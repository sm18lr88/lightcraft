//! Executes an ONNX graph's operators on candle: the operators of PaddleOCR's exported PP-OCRv6 detector and
//! recogniser and its PP-LCNet text-line orientation classifier (opsets 7 to 17).
//!
//! Convolutions (`auto_pad` SAME_UPPER, unequal strides, groups), transposed convolutions, BatchNormalization,
//! the activations (Relu, Sigmoid, HardSigmoid, Erf), element-wise arithmetic with broadcasting (Add, Sub, Mul, Div,
//! Pow, Sqrt), ReduceMean, Softmax, MatMul, pooling, nearest Resize, the layout operators (Transpose, Reshape, Squeeze,
//! Unsqueeze, Concat, Slice) and the shape arithmetic (Shape, Gather, Slice and Concat on int64 vectors) that feeds
//! Reshape its sizes; Constant nodes become constants at load. Unsupported operators, attributes and dangling value
//! names fail at load, not in the middle of a run.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use candle_core::{Device, Tensor};

use crate::onnx::{self, Attr, Data, ModelProto, NodeProto, TensorProto};
use crate::{Error, Result};

/// Largest tensor a shape computed at run time may ask for, in elements (1 GB of f32).
const MAX_RUNTIME_ELEMENTS: usize = 1 << 28;
/// Largest dilation and padding a window may have (PP-OCR's use a dilation of 1 and pads of a few pixels): a model file
/// asking for more is refused at load, before any arithmetic on it.
const MAX_DILATION: usize = 64;
const MAX_PAD: usize = 1024;
/// Largest kernel side a window may have (PP-OCR's are a few taps): a model file asking for more is refused at load,
/// before a pad of half that size is allocated.
const MAX_KERNEL: usize = 1024;

/// The default-domain operator sets whose semantics for these operators [`Graph`] follows.
pub const OPSETS: std::ops::RangeInclusive<i64> = 7..=17;

#[derive(Clone, Debug)]
enum Value {
    F(Tensor),
    I(Vec<i64>),
}

/// Padding of a 2-D window: explicit (top, left, bottom, right) or ONNX's SAME_UPPER (worked out per input size).
#[derive(Clone, Copy, Debug)]
enum Pads {
    Fixed([usize; 4]),
    SameUpper,
}

#[derive(Clone, Copy, Debug)]
struct Window {
    kernel: [usize; 2],
    strides: [usize; 2],
    dilation: usize,
    pads: Pads,
}

#[derive(Clone, Debug)]
enum Op {
    Conv {
        window: Window,
        groups: usize,
    },
    ConvTranspose {
        stride: usize,
    },
    Relu,
    Sigmoid,
    HardSigmoid {
        alpha: f64,
        beta: f64,
    },
    Erf,
    Sqrt,
    Add,
    Sub,
    Mul,
    Div,
    /// `exponent`: the exponent when it is a constant (read once, at load).
    Pow {
        exponent: Option<f32>,
    },
    /// Dropout and Identity: the input, unchanged (inference).
    Pass,
    GlobalAveragePool,
    BatchNorm {
        eps: f64,
    },
    MaxPool {
        window: Window,
        ceil: bool,
    },
    AveragePool {
        window: Window,
        ceil: bool,
    },
    ReduceMean {
        axes: Vec<i64>,
        keep: bool,
    },
    /// Before opset 13 Softmax flattens the dimensions from `axis` on into one; from 13 it is over `axis` alone.
    Softmax {
        axis: i64,
        flatten: bool,
    },
    MatMul,
    Shape,
    Gather,
    Unsqueeze {
        axes: Vec<i64>,
    },
    Squeeze {
        axes: Vec<i64>,
    },
    /// Opset 10+: starts, ends, axes, steps as inputs. Opset 7: as attributes.
    Slice {
        attrs: Option<(Vec<i64>, Vec<i64>, Vec<i64>)>,
    },
    Concat {
        axis: i64,
    },
    Resize,
    Transpose {
        perm: Option<Vec<usize>>,
    },
    Reshape,
}

struct Node {
    name: String,
    op_type: String,
    op: Op,
    inputs: Vec<String>,
    output: String,
    /// Values no later node reads, dropped after this one.
    free: Vec<String>,
}

/// A loaded graph, ready to run.
pub struct Graph {
    nodes: Vec<Node>,
    consts: HashMap<String, Value>,
    inputs: Vec<String>,
    outputs: Vec<String>,
    device: Device,
}

fn msg(m: impl Into<String>) -> Error {
    Error::Model(m.into())
}

fn unsupported(n: &NodeProto, what: impl std::fmt::Display) -> Error {
    msg(format!("node {} ({}): {what}", n.name, n.op_type))
}

fn attr_int(n: &NodeProto, name: &str, default: i64) -> Result<i64> {
    match n.attr(name) {
        None => Ok(default),
        Some(Attr::Int(v)) => Ok(*v),
        Some(_) => Err(unsupported(n, format!("attribute {name} is not an integer"))),
    }
}

fn attr_float(n: &NodeProto, name: &str, default: f32) -> Result<f32> {
    match n.attr(name) {
        None => Ok(default),
        Some(Attr::Float(v)) => Ok(*v),
        Some(_) => Err(unsupported(n, format!("attribute {name} is not a float"))),
    }
}

fn attr_ints(n: &NodeProto, name: &str) -> Result<Option<Vec<i64>>> {
    match n.attr(name) {
        None => Ok(None),
        Some(Attr::Ints(v)) => Ok(Some(v.clone())),
        Some(_) => Err(unsupported(n, format!("attribute {name} is not a list of integers"))),
    }
}

fn attr_str(n: &NodeProto, name: &str, default: &str) -> Result<String> {
    match n.attr(name) {
        None => Ok(default.to_string()),
        Some(Attr::Bytes(v)) => Ok(String::from_utf8_lossy(v).into_owned()),
        Some(_) => Err(unsupported(n, format!("attribute {name} is not a string"))),
    }
}

fn usizes(n: &NodeProto, name: &str, v: &[i64]) -> Result<Vec<usize>> {
    v.iter().map(|x| usize::try_from(*x).map_err(|_| unsupported(n, format!("negative {name} {x}")))).collect()
}

fn pair(n: &NodeProto, name: &str, default: usize) -> Result<[usize; 2]> {
    let v = usizes(n, name, &attr_ints(n, name)?.unwrap_or_else(|| vec![default as i64; 2]))?;
    match v[..] {
        [a, b] if a > 0 && b > 0 => Ok([a, b]),
        _ => Err(unsupported(n, format!("{name} {v:?} (needs two positive values)"))),
    }
}

/// A 2-D window's kernel, strides, dilation and padding. `kernel` comes from the weights for a convolution.
fn window(n: &NodeProto, kernel: Option<[usize; 2]>) -> Result<Window> {
    let kernel = match kernel {
        Some(k) => k,
        None => pair(n, "kernel_shape", 1)?,
    };
    if kernel.contains(&0) {
        return Err(unsupported(n, format!("an empty kernel {kernel:?}")));
    }
    if kernel.iter().any(|k| *k > MAX_KERNEL) {
        return Err(unsupported(n, format!("kernel {kernel:?} (at most {MAX_KERNEL})")));
    }
    let [dh, dw] = pair(n, "dilations", 1)?;
    if dh != dw {
        return Err(unsupported(n, "unequal dilations"));
    }
    if dh > MAX_DILATION {
        return Err(unsupported(n, format!("dilation {dh} (at most {MAX_DILATION})")));
    }
    let pads = match attr_str(n, "auto_pad", "NOTSET")?.as_str() {
        "NOTSET" => {
            let p = usizes(n, "pads", &attr_ints(n, "pads")?.unwrap_or_else(|| vec![0; 4]))?;
            let [t, l, b, r] = p[..] else { return Err(unsupported(n, "only 2-D windows are supported")) };
            if p.iter().any(|v| *v > MAX_PAD) {
                return Err(unsupported(n, format!("pads {p:?} (at most {MAX_PAD})")));
            }
            Pads::Fixed([t, l, b, r])
        }
        "SAME_UPPER" => Pads::SameUpper,
        "VALID" => Pads::Fixed([0; 4]),
        other => return Err(unsupported(n, format!("auto_pad {other}"))),
    };
    Ok(Window { kernel, strides: pair(n, "strides", 1)?, dilation: dh, pads })
}

/// The extent of a kernel of `k` taps `dilation` apart: `(k - 1) · dilation + 1`, checked.
fn span(k: usize, dilation: usize) -> Result<usize> {
    k.checked_sub(1)
        .and_then(|k| k.checked_mul(dilation))
        .and_then(|e| e.checked_add(1))
        .ok_or_else(|| msg(format!("a window of {k} taps with dilation {dilation}")))
}

impl Window {
    /// (top, left, bottom, right) for an input of `h` × `w`.
    fn pads(&self, h: usize, w: usize) -> Result<[usize; 4]> {
        match self.pads {
            Pads::Fixed(p) => Ok(p),
            Pads::SameUpper => {
                let total = |len: usize, k: usize, s: usize| -> Result<usize> {
                    let out = len.div_ceil(s.max(1));
                    let span = span(k, self.dilation)?;
                    let reach = out.saturating_sub(1).checked_mul(s).and_then(|v| v.checked_add(span)).ok_or_else(|| msg("a window too large"))?;
                    Ok(reach.saturating_sub(len))
                };
                let ([kh, kw], [sh, sw]) = (self.kernel, self.strides);
                let (th, tw) = (total(h, kh, sh)?, total(w, kw, sw)?);
                Ok([th / 2, tw / 2, th - th / 2, tw - tw / 2])
            }
        }
    }
}

fn compile(n: &NodeProto, consts: &HashMap<String, Value>, opset: i64) -> Result<Op> {
    if !(n.domain.is_empty() || n.domain == "ai.onnx") {
        return Err(unsupported(n, format!("operator domain {:?} is not supported", n.domain)));
    }
    Ok(match n.op_type.as_str() {
        "Conv" => {
            let w = n.inputs.get(1).and_then(|w| consts.get(w));
            let kernel = match w {
                Some(Value::F(t)) if t.rank() == 4 => Some([t.dim(2)?, t.dim(3)?]),
                _ => return Err(unsupported(n, "the weights must be a constant 4-D tensor")),
            };
            let groups = usize::try_from(attr_int(n, "group", 1)?).ok().filter(|g| *g > 0).ok_or_else(|| unsupported(n, "bad group"))?;
            Op::Conv { window: window(n, kernel)?, groups }
        }
        "ConvTranspose" => {
            let win = window(n, None)?;
            let [sh, sw] = win.strides;
            if sh != sw || win.dilation != 1 || attr_int(n, "group", 1)? != 1 || !matches!(win.pads, Pads::Fixed([0, 0, 0, 0])) {
                return Err(unsupported(n, "only an unpadded, ungrouped transposed convolution with equal strides is supported"));
            }
            if attr_ints(n, "output_padding")?.is_some_and(|p| p.iter().any(|v| *v != 0)) || n.attr("output_shape").is_some() {
                return Err(unsupported(n, "output_padding / output_shape"));
            }
            Op::ConvTranspose { stride: sh }
        }
        "Relu" => Op::Relu,
        "Sigmoid" => Op::Sigmoid,
        "HardSigmoid" => Op::HardSigmoid { alpha: f64::from(attr_float(n, "alpha", 0.2)?), beta: f64::from(attr_float(n, "beta", 0.5)?) },
        "Erf" => Op::Erf,
        "Sqrt" => Op::Sqrt,
        "Add" => Op::Add,
        "Sub" => Op::Sub,
        "Mul" => Op::Mul,
        "Div" => Op::Div,
        "Pow" => Op::Pow {
            exponent: match n.inputs.get(1).and_then(|e| consts.get(e)) {
                Some(Value::F(t)) if t.elem_count() == 1 => t.flatten_all()?.to_vec1::<f32>()?.first().copied(),
                _ => None,
            },
        },
        "Identity" => Op::Pass,
        "Dropout" => {
            if attr_int(n, "training_mode", 0)? != 0 {
                return Err(unsupported(n, "training mode"));
            }
            Op::Pass
        }
        "GlobalAveragePool" => Op::GlobalAveragePool,
        "BatchNormalization" => {
            if attr_int(n, "training_mode", 0)? != 0 || attr_int(n, "spatial", 1)? != 1 {
                return Err(unsupported(n, "training mode or non-spatial"));
            }
            Op::BatchNorm { eps: f64::from(attr_float(n, "epsilon", 1e-5)?) }
        }
        "MaxPool" | "AveragePool" => {
            if attr_int(n, "storage_order", 0)? != 0 {
                return Err(unsupported(n, "storage_order"));
            }
            let win = window(n, None)?;
            let ceil = attr_int(n, "ceil_mode", 0)? != 0;
            if n.op_type == "MaxPool" {
                Op::MaxPool { window: win, ceil }
            } else {
                // padding that counts would change the averages at the borders
                if attr_int(n, "count_include_pad", 0)? != 0 && !matches!(win.pads, Pads::Fixed([0, 0, 0, 0])) {
                    return Err(unsupported(n, "count_include_pad with padding"));
                }
                if !matches!(win.pads, Pads::Fixed([0, 0, 0, 0])) {
                    return Err(unsupported(n, "padded average pooling"));
                }
                Op::AveragePool { window: win, ceil }
            }
        }
        "ReduceMean" => {
            if n.inputs.len() > 1 {
                return Err(unsupported(n, "axes as an input (opset 18) are not supported"));
            }
            Op::ReduceMean {
                axes: attr_ints(n, "axes")?.ok_or_else(|| unsupported(n, "reducing over every axis is not supported"))?,
                keep: attr_int(n, "keepdims", 1)? != 0,
            }
        }
        "Softmax" => Op::Softmax { axis: attr_int(n, "axis", if opset >= 13 { -1 } else { 1 })?, flatten: opset < 13 },
        "MatMul" => Op::MatMul,
        "Shape" => {
            // opset 15's start / end would slice the dims; the run always returns every one
            if n.attr("start").is_some() || n.attr("end").is_some() {
                return Err(unsupported(n, "Shape start / end (opset 15) are not supported"));
            }
            Op::Shape
        }
        "Gather" => {
            if attr_int(n, "axis", 0)? != 0 {
                return Err(unsupported(n, "Gather is only supported on axis 0 of a shape"));
            }
            Op::Gather
        }
        "Unsqueeze" => Op::Unsqueeze { axes: attr_ints(n, "axes")?.ok_or_else(|| unsupported(n, "axes as an input (opset 13) are not supported"))? },
        "Squeeze" => Op::Squeeze { axes: attr_ints(n, "axes")?.ok_or_else(|| unsupported(n, "Squeeze needs its axes as an attribute"))? },
        "Slice" => {
            if opset < 10 {
                let starts = attr_ints(n, "starts")?.ok_or_else(|| unsupported(n, "no starts"))?;
                let ends = attr_ints(n, "ends")?.ok_or_else(|| unsupported(n, "no ends"))?;
                let axes = attr_ints(n, "axes")?.unwrap_or_else(|| (0..starts.len() as i64).collect());
                Op::Slice { attrs: Some((starts, ends, axes)) }
            } else {
                Op::Slice { attrs: None }
            }
        }
        "Concat" => Op::Concat { axis: attr_int(n, "axis", i64::MIN).ok().filter(|a| *a != i64::MIN).ok_or_else(|| unsupported(n, "no axis"))? },
        "Resize" => {
            let mode = attr_str(n, "mode", "nearest")?;
            let ctm = attr_str(n, "coordinate_transformation_mode", "half_pixel")?;
            let nearest = attr_str(n, "nearest_mode", "round_prefer_floor")?;
            if (mode.as_str(), ctm.as_str(), nearest.as_str()) != ("nearest", "asymmetric", "floor") {
                return Err(unsupported(n, format!("Resize {mode}/{ctm}/{nearest} (only nearest/asymmetric/floor)")));
            }
            Op::Resize
        }
        "Transpose" => Op::Transpose { perm: attr_ints(n, "perm")?.map(|p| usizes(n, "perm", &p)).transpose()? },
        "Reshape" => {
            if attr_int(n, "allowzero", 0)? != 0 {
                return Err(unsupported(n, "allowzero"));
            }
            Op::Reshape
        }
        _ => return Err(unsupported(n, "operator not supported")),
    })
}

fn constant(t: TensorProto, device: &Device) -> Result<Value> {
    Ok(match t.data {
        Data::F32(v) => Value::F(Tensor::from_vec(v, t.dims, device)?),
        Data::I64(v) => Value::I(v),
    })
}

impl Graph {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let model = onnx::read(path)?;
        Self::new(model, device).map_err(|e| match e {
            Error::Model(m) => msg(format!("{}: {m}", path.display())),
            e => e,
        })
    }

    pub fn new(model: ModelProto, device: &Device) -> Result<Self> {
        let opset = match model.opset {
            Some(v) if OPSETS.contains(&v) => v,
            other => return Err(msg(format!("operator set {other:?} is not supported (needs {} to {})", OPSETS.start(), OPSETS.end()))),
        };
        let mut consts = HashMap::new();
        for t in model.initializers {
            let name = t.name.clone();
            consts.insert(name, constant(t, device)?);
        }
        let inputs: Vec<String> = model.inputs.into_iter().filter(|i| !consts.contains_key(i)).collect();
        // every value must be computed before it is used
        let mut known: HashSet<String> = consts.keys().cloned().chain(inputs.iter().cloned()).collect();
        let mut nodes = Vec::with_capacity(model.nodes.len());
        for n in &model.nodes {
            for i in &n.inputs {
                if !i.is_empty() && !known.contains(i) {
                    return Err(unsupported(n, format!("uses {i:?} before it is computed")));
                }
            }
            let output = n.outputs.first().filter(|o| !o.is_empty()).ok_or_else(|| unsupported(n, "no output"))?;
            if n.op_type == "Constant" {
                let Some(Attr::Tensor(t)) = n.attr("value") else { return Err(unsupported(n, "only a tensor `value` is supported")) };
                consts.insert(output.clone(), constant(t.clone(), device)?);
                known.insert(output.clone());
                continue;
            }
            // since opset 12 Dropout's training mode is its third input: only a constant false one is inference
            if n.op_type == "Dropout"
                && let Some(t) = n.inputs.get(2).filter(|t| !t.is_empty())
                && !matches!(consts.get(t.as_str()), Some(Value::I(v)) if v.iter().all(|x| *x == 0))
            {
                return Err(unsupported(n, "training mode"));
            }
            known.insert(output.clone());
            nodes.push(Node {
                name: n.name.clone(),
                op_type: n.op_type.clone(),
                op: compile(n, &consts, opset)?,
                inputs: n.inputs.clone(),
                output: output.clone(),
                free: Vec::new(),
            });
        }
        for o in &model.outputs {
            if !known.contains(o) {
                return Err(msg(format!("graph output {o:?} is never computed")));
            }
        }
        // drop each intermediate value after its last reader
        let mut last: HashMap<&str, usize> = HashMap::new();
        for (i, n) in nodes.iter().enumerate() {
            for name in n.inputs.iter().filter(|s| !s.is_empty()) {
                last.insert(name.as_str(), i);
            }
        }
        let keep: HashSet<&str> = model.outputs.iter().map(String::as_str).chain(consts.keys().map(String::as_str)).collect();
        let free: Vec<(usize, String)> = last.into_iter().filter(|(n, _)| !keep.contains(n)).map(|(n, i)| (i, n.to_string())).collect();
        for (i, name) in free {
            if let Some(node) = nodes.get_mut(i) {
                node.free.push(name);
            }
        }
        Ok(Self { nodes, consts, inputs, outputs: model.outputs, device: device.clone() })
    }

    /// The device the graph's weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Runs the graph on its single input. The outputs, in the graph's order.
    pub fn run(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let [name] = &self.inputs[..] else { return Err(msg(format!("the model has {} inputs, not one", self.inputs.len()))) };
        let mut env: HashMap<&str, Value> = HashMap::new();
        env.insert(name.as_str(), Value::F(x.to_device(&self.device)?));
        // LIGHTCRAFT_PROFILE=ops: the time of each operator type, waiting for the device after each node
        let profile = std::env::var("LIGHTCRAFT_PROFILE").is_ok_and(|v| v == "ops");
        let mut spent: HashMap<&str, (f64, usize)> = HashMap::new();
        for node in &self.nodes {
            let started = std::time::Instant::now();
            let out = {
                let mut args = Vec::with_capacity(node.inputs.len());
                for name in &node.inputs {
                    args.push(if name.is_empty() {
                        None
                    } else {
                        Some(env.get(name.as_str()).or_else(|| self.consts.get(name)).ok_or_else(|| msg(format!("value {name:?} is missing")))?)
                    });
                }
                exec(&node.op, &args).map_err(|e| {
                    let m = match e {
                        Error::Model(m) | Error::Input(m) => m,
                        other => other.to_string(),
                    };
                    msg(format!("node {} ({}): {m}", node.name, node.op_type))
                })?
            };
            if profile {
                self.device.synchronize()?;
                let e = spent.entry(node.op_type.as_str()).or_default();
                e.0 += started.elapsed().as_secs_f64() * 1000.0;
                e.1 += 1;
            }
            for f in &node.free {
                env.remove(f.as_str());
            }
            env.insert(node.output.as_str(), out);
        }
        if profile {
            let mut rows: Vec<_> = spent.into_iter().collect();
            rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
            let table: Vec<String> = rows.iter().take(8).map(|(op, (ms, n))| format!("{op} {ms:.1} ms ({n})")).collect();
            eprintln!("graph {:?} on {:?}: {}", x.dims(), self.device.location(), table.join(", "));
        }
        self.outputs
            .iter()
            .map(|o| match env.remove(o.as_str()).or_else(|| self.consts.get(o).cloned()) {
                Some(Value::F(t)) => Ok(t),
                Some(Value::I(_)) => Err(msg(format!("output {o:?} is not a float tensor"))),
                None => Err(msg(format!("output {o:?} was not computed"))),
            })
            .collect()
    }
}

fn arg<'a>(a: &[Option<&'a Value>], i: usize) -> Option<&'a Value> {
    a.get(i).copied().flatten()
}

fn tensor<'a>(a: &[Option<&'a Value>], i: usize) -> Result<&'a Tensor> {
    match arg(a, i) {
        Some(Value::F(t)) => Ok(t),
        Some(Value::I(_)) => Err(msg(format!("input {i} must be a float tensor"))),
        None => Err(msg(format!("input {i} is missing"))),
    }
}

fn ints<'a>(a: &[Option<&'a Value>], i: usize) -> Result<&'a [i64]> {
    match arg(a, i) {
        Some(Value::I(v)) => Ok(v),
        Some(Value::F(_)) => Err(msg(format!("input {i} must be an int64 tensor"))),
        None => Err(msg(format!("input {i} is missing"))),
    }
}

/// `t` with leading 1s up to rank `r` (numpy-style broadcasting aligns trailing dims).
fn lift(t: &Tensor, r: usize) -> Result<Tensor> {
    if t.rank() >= r {
        return Ok(t.clone());
    }
    let mut d = vec![1; r - t.rank()];
    d.extend_from_slice(t.dims());
    Ok(t.reshape(d)?)
}

fn broadcast(a: &Tensor, b: &Tensor) -> Result<(Tensor, Tensor)> {
    let r = a.rank().max(b.rank());
    Ok((lift(a, r)?, lift(b, r)?))
}

/// `axis` (negative counts from the end) as an index into `rank` dims.
fn axis(axis: i64, rank: usize) -> Result<usize> {
    let r = i64::try_from(rank).map_err(|_| msg("rank"))?;
    let a = if axis < 0 { axis.saturating_add(r) } else { axis };
    usize::try_from(a).ok().filter(|a| *a < rank).ok_or_else(|| msg(format!("axis {axis} out of range for rank {rank}")))
}

/// ONNX Slice bounds on a dimension of `len`: negatives count from the end, then clamped.
fn slice_range(start: i64, end: i64, len: usize) -> (usize, usize) {
    let n = i64::try_from(len).unwrap_or(i64::MAX);
    let fix = |v: i64| (if v < 0 { v.saturating_add(n) } else { v }).clamp(0, n);
    let (s, e) = (fix(start), fix(end));
    (s as usize, e.max(s) as usize)
}

/// `x` in a layout candle's CPU conv2d reads correctly. Its tiled im2col path assumes an input with strides
/// `[h·w·c, w·c, c, 1]` is stored channels-last; a contiguous NCHW tensor with c = h = w has exactly those strides and
/// would be misread. The same values with channels-last storage can't be mistaken.
fn conv_safe_layout(x: &Tensor) -> Result<Tensor> {
    let (_, c, h, w) = x.dims4()?;
    if x.device().is_cpu() && c > 1 && x.stride() == [h * w * c, w * c, c, 1] {
        return Ok(x.permute((0, 2, 3, 1))?.contiguous()?.permute((0, 3, 1, 2))?);
    }
    Ok(x.clone())
}

fn checked_elements(dims: &[usize]) -> Result<usize> {
    dims.iter()
        .try_fold(1usize, |a, d| a.checked_mul(*d))
        .filter(|n| *n <= MAX_RUNTIME_ELEMENTS)
        .ok_or_else(|| msg(format!("a tensor of {dims:?} is too large")))
}

/// Every `step`-th index of `dim` from 0 (a stride candle's ops can't take directly).
fn every(x: &Tensor, dim: usize, step: usize, count: usize) -> Result<Tensor> {
    if step == 1 && x.dim(dim)? == count {
        return Ok(x.clone());
    }
    let idx: Vec<u32> = (0..count).map(|i| (i * step) as u32).collect();
    Ok(x.index_select(&Tensor::from_vec(idx, count, x.device())?, dim)?)
}

/// Output length of a window over `len` (padded) values.
fn out_len(len: usize, k: usize, s: usize, ceil: bool) -> Result<usize> {
    if len < k {
        return Err(msg(format!("a feature map of {len} is smaller than the window of {k}")));
    }
    let span = len - k;
    Ok(if ceil { span.div_ceil(s) } else { span / s } + 1)
}

fn pool(x: &Tensor, win: &Window, ceil: bool, max: bool) -> Result<Tensor> {
    let (n, c, h, w) = x.dims4()?;
    let [t, l, b, r] = win.pads(h, w)?;
    checked_elements(&[n, c, h.saturating_add(t).saturating_add(b), w.saturating_add(l).saturating_add(r)])?;
    // max pooling: padding with the edge values never wins over a real value, as ONNX's -inf padding
    let x = if t + l + b + r > 0 { x.pad_with_same(2, t, b)?.pad_with_same(3, l, r)? } else { x.clone() };
    let (_, _, h, w) = x.dims4()?;
    let ([kh, kw], [sh, sw]) = (win.kernel, win.strides);
    if win.dilation != 1 {
        return Err(msg("dilated pooling is not supported"));
    }
    let (oh, ow) = (out_len(h, kh, sh, ceil)?, out_len(w, kw, sw, ceil)?);
    // ceil mode: the last window may run past the edge; extend with edge values (max) — averages are refused there
    let need_h = (oh - 1) * sh + kh;
    let need_w = (ow - 1) * sw + kw;
    let x = if need_h > h || need_w > w {
        if !max {
            return Err(msg("ceil-mode average pooling over a partial window is not supported"));
        }
        x.pad_with_same(2, 0, need_h.saturating_sub(h))?.pad_with_same(3, 0, need_w.saturating_sub(w))?
    } else {
        x
    };
    // stride 1 windows, then every stride-th one (candle's pooling takes one stride per axis but equal kernels)
    let y = if max { x.max_pool2d_with_stride((kh, kw), (sh, sw))? } else { x.avg_pool2d_with_stride((kh, kw), (sh, sw))? };
    Ok(y.narrow(2, 0, oh.min(y.dim(2)?))?.narrow(3, 0, ow.min(y.dim(3)?))?)
}

/// A depthwise convolution (one filter per channel) as the sum of its kernel's taps: each tap is the input shifted by
/// the tap's offset times that tap's per-channel weight. candle runs a grouped convolution as one convolution per group
/// (hundreds of small kernels for a depthwise layer, very slow on a graphics card); this is `kh × kw` whole-tensor
/// operations instead.
fn depthwise(x: &Tensor, w: &Tensor, pad: usize, [sh, sw]: [usize; 2], dilation: usize) -> Result<Tensor> {
    let x = if pad > 0 { x.pad_with_zeros(2, pad, pad)?.pad_with_zeros(3, pad, pad)? } else { x.clone() };
    let (_, c, h, wd) = x.dims4()?;
    let (_, _, kh, kw) = w.dims4()?;
    if kh == 0 || kw == 0 || dilation == 0 {
        return Err(msg("an empty kernel"));
    }
    let (eh, ew) = ((kh - 1) * dilation + 1, (kw - 1) * dilation + 1);
    if h < eh || wd < ew || sh == 0 || sw == 0 {
        return Err(msg(format!("a feature map of {h}×{wd} is smaller than the window of {eh}×{ew}")));
    }
    let (oh, ow) = ((h - eh) / sh + 1, (wd - ew) / sw + 1);
    let taps = w.reshape((c, kh * kw))?.t()?.contiguous()?;
    // with a stride, the input's sh × sw phases (every sh-th row from row pi, every sw-th column from column pj), each
    // made once: a tap at offset (oi, oj) reads phase (oi % sh, oj % sw) from (oi / sh, oj / sw)
    let mut phases: Vec<Option<Tensor>> = vec![None; sh * sw];
    let mut acc: Option<Tensor> = None;
    for i in 0..kh {
        for j in 0..kw {
            let (oi, oj) = (i * dilation, j * dilation);
            let (pi, pj) = (oi % sh, oj % sw);
            let slot = phases.get_mut(pi * sw + pj).ok_or_else(|| msg("phase"))?;
            let phase = match slot {
                Some(p) => p.clone(),
                None => {
                    let v = x.narrow(2, pi, h - pi)?.narrow(3, pj, wd - pj)?.contiguous()?;
                    let p = every(&every(&v, 2, sh, (h - pi).div_ceil(sh))?, 3, sw, (wd - pj).div_ceil(sw))?;
                    slot.insert(p).clone()
                }
            };
            let xs = phase.narrow(2, oi / sh, oh)?.narrow(3, oj / sw, ow)?;
            let tap = taps.get(i * kw + j)?.reshape((1, c, 1, 1))?;
            let term = xs.broadcast_mul(&tap)?;
            acc = Some(match acc {
                Some(a) => (a + term)?,
                None => term,
            });
        }
    }
    acc.ok_or_else(|| msg("an empty kernel"))
}

fn exec(op: &Op, a: &[Option<&Value>]) -> Result<Value> {
    let out = match op {
        Op::Conv { window: win, groups } => {
            let (x, w) = (tensor(a, 0)?, tensor(a, 1)?);
            if x.rank() != 4 || w.rank() != 4 {
                return Err(msg("only 2-D convolutions are supported"));
            }
            let (batch, chans, h, wd) = x.dims4()?;
            let [t, l, b, r] = win.pads(h, wd)?;
            let (c_out, c_in_k, kh, kw) = w.dims4()?;
            // the padded input must hold the window (candle's output-size arithmetic is not checked)
            let (eh, ew) = (span(kh, win.dilation)?, span(kw, win.dilation)?);
            let (ph, pw) = (h.saturating_add(t).saturating_add(b), wd.saturating_add(l).saturating_add(r));
            if ph < eh || pw < ew {
                return Err(msg(format!("a feature map of {ph}×{pw} is smaller than the window of {eh}×{ew}")));
            }
            checked_elements(&[batch, chans, ph, pw])?;
            let (x, p) = if t == l && l == b && b == r { (x.clone(), t) } else { (x.pad_with_zeros(2, t, b)?.pad_with_zeros(3, l, r)?, 0) };
            let [sh, sw] = win.strides;
            let y = if *groups > 1 && c_in_k == 1 && c_out == *groups && x.dim(1)? == *groups {
                depthwise(&x, w, p, [sh, sw], win.dilation)?
            } else if *groups == 1 && (kh, kw, sh, sw, p) == (1, 1, 1, 1, 0) {
                // a pointwise convolution is one matrix product ([c_out, c_in] × [c_in, h·w] per image), not im2col
                let (n, c, h, wd) = x.dims4()?;
                let y = w.reshape((c_out, c))?.broadcast_matmul(&x.reshape((n, c, h * wd))?)?;
                y.reshape((n, c_out, h, wd))?
            } else if sh == sw {
                conv_safe_layout(&x)?.conv2d(w, p, sh, win.dilation, *groups)?
            } else {
                // unequal strides: the stride-1 convolution, then every sh-th row and sw-th column
                let y = conv_safe_layout(&x)?.conv2d(w, p, 1, win.dilation, *groups)?;
                let (_, _, yh, yw) = y.dims4()?;
                let y = every(&y, 2, sh, (yh - 1) / sh + 1)?;
                every(&y, 3, sw, (yw - 1) / sw + 1)?
            };
            match arg(a, 2) {
                Some(_) => {
                    let bias = tensor(a, 2)?;
                    y.broadcast_add(&bias.reshape((1, bias.elem_count(), 1, 1))?)?
                }
                None => y,
            }
        }
        Op::ConvTranspose { stride } => {
            let (x, w) = (tensor(a, 0)?, tensor(a, 1)?);
            let y = x.contiguous()?.conv_transpose2d(&w.contiguous()?, 0, 0, *stride, 1)?;
            match arg(a, 2) {
                Some(_) => {
                    let bias = tensor(a, 2)?;
                    y.broadcast_add(&bias.reshape((1, bias.elem_count(), 1, 1))?)?
                }
                None => y,
            }
        }
        Op::Relu => tensor(a, 0)?.relu()?,
        Op::Sigmoid => candle_nn::ops::sigmoid(tensor(a, 0)?)?,
        Op::HardSigmoid { alpha, beta } => tensor(a, 0)?.affine(*alpha, *beta)?.clamp(0f32, 1f32)?,
        Op::Erf => tensor(a, 0)?.erf()?,
        Op::Sqrt => tensor(a, 0)?.sqrt()?,
        Op::Add => {
            let (x, y) = broadcast(tensor(a, 0)?, tensor(a, 1)?)?;
            x.broadcast_add(&y)?
        }
        Op::Sub => {
            let (x, y) = broadcast(tensor(a, 0)?, tensor(a, 1)?)?;
            x.broadcast_sub(&y)?
        }
        Op::Mul => {
            let (x, y) = broadcast(tensor(a, 0)?, tensor(a, 1)?)?;
            x.broadcast_mul(&y)?
        }
        Op::Div => {
            let (x, y) = broadcast(tensor(a, 0)?, tensor(a, 1)?)?;
            x.broadcast_div(&y)?
        }
        Op::Pow { exponent } => {
            let (x, e) = (tensor(a, 0)?, tensor(a, 1)?);
            // the graphs square (layer norm's variance); a general power goes through exp/log. A constant exponent was
            // read at load (no copy from the device per run).
            let scalar = match exponent {
                Some(p) => vec![*p],
                None => e.flatten_all()?.to_vec1::<f32>()?,
            };
            match scalar[..] {
                [2.0] => x.sqr()?,
                [1.0] => x.clone(),
                [0.5] => x.sqrt()?,
                [p] => x.powf(f64::from(p))?,
                _ => {
                    let (x, e) = broadcast(x, e)?;
                    x.broadcast_pow(&e)?
                }
            }
        }
        Op::Pass => match arg(a, 0) {
            Some(Value::I(v)) => return Ok(Value::I(v.clone())),
            _ => tensor(a, 0)?.clone(),
        },
        Op::GlobalAveragePool => {
            let x = tensor(a, 0)?;
            if x.rank() != 4 {
                return Err(msg(format!("GlobalAveragePool takes [N, C, H, W], got {:?}", x.dims())));
            }
            x.mean_keepdim(3)?.mean_keepdim(2)?
        }
        Op::BatchNorm { eps } => {
            let x = tensor(a, 0)?;
            let (scale, bias, mean, var) = (tensor(a, 1)?, tensor(a, 2)?, tensor(a, 3)?, tensor(a, 4)?);
            let mul = (scale / (var + *eps)?.sqrt()?)?;
            let add = (bias - mean.mul(&mul)?)?;
            let mut d = vec![1; x.rank()];
            *d.get_mut(1).ok_or_else(|| msg("BatchNormalization needs a channel dimension"))? = mul.elem_count();
            x.broadcast_mul(&mul.reshape(d.clone())?)?.broadcast_add(&add.reshape(d)?)?
        }
        Op::MaxPool { window: win, ceil } => pool(tensor(a, 0)?, win, *ceil, true)?,
        Op::AveragePool { window: win, ceil } => pool(tensor(a, 0)?, win, *ceil, false)?,
        Op::ReduceMean { axes, keep } => {
            let x = tensor(a, 0)?;
            let rank = x.rank();
            let mut ax = axes.iter().map(|v| axis(*v, rank)).collect::<Result<Vec<_>>>()?;
            ax.sort_unstable();
            ax.dedup();
            let mut y = x.clone();
            for d in ax.iter().rev() {
                y = if *keep { y.mean_keepdim(*d)? } else { y.mean(*d)? };
            }
            y
        }
        Op::Softmax { axis: ax, flatten } => {
            let x = tensor(a, 0)?;
            let d = axis(*ax, x.rank())?;
            if *flatten && d + 1 != x.rank() {
                let dims = x.dims().to_vec();
                let (lo, hi) = dims.split_at_checked(d).ok_or_else(|| msg("axis out of range"))?;
                let y = x.reshape((lo.iter().product::<usize>(), hi.iter().product::<usize>()))?;
                candle_nn::ops::softmax_last_dim(&y.contiguous()?)?.reshape(dims)?
            } else if d + 1 == x.rank() {
                candle_nn::ops::softmax_last_dim(&x.contiguous()?)?
            } else {
                candle_nn::ops::softmax(x, d)?
            }
        }
        Op::MatMul => {
            let (x, y) = (tensor(a, 0)?, tensor(a, 1)?);
            if x.rank() < 2 || y.rank() < 2 {
                return Err(msg("MatMul of vectors is not supported"));
            }
            x.contiguous()?.broadcast_matmul(&y.contiguous()?)?
        }
        Op::Shape => {
            return Ok(Value::I(match arg(a, 0) {
                Some(Value::F(t)) => t.dims().iter().map(|d| *d as i64).collect(),
                Some(Value::I(v)) => vec![v.len() as i64],
                None => return Err(msg("input 0 is missing")),
            }));
        }
        Op::Gather => {
            let (data, idx) = (ints(a, 0)?, ints(a, 1)?);
            let n = data.len() as i64;
            let v = idx
                .iter()
                .map(|i| {
                    let j = if *i < 0 { i.saturating_add(n) } else { *i };
                    usize::try_from(j).ok().and_then(|j| data.get(j)).copied().ok_or_else(|| msg(format!("index {i} out of range")))
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Value::I(v));
        }
        Op::Unsqueeze { axes } => match arg(a, 0) {
            Some(Value::I(v)) => return Ok(Value::I(v.clone())),
            _ => {
                let mut t = tensor(a, 0)?.clone();
                let rank = t.rank().saturating_add(axes.len());
                let mut ax = axes.iter().map(|x| axis(*x, rank)).collect::<Result<Vec<_>>>()?;
                ax.sort_unstable();
                for x in ax {
                    t = t.unsqueeze(x)?;
                }
                t
            }
        },
        Op::Squeeze { axes } => match arg(a, 0) {
            Some(Value::I(v)) => return Ok(Value::I(v.clone())),
            _ => {
                let mut t = tensor(a, 0)?.clone();
                let mut ax = axes.iter().map(|x| axis(*x, t.rank())).collect::<Result<Vec<_>>>()?;
                ax.sort_unstable();
                ax.dedup();
                for x in ax.into_iter().rev() {
                    if t.dim(x)? != 1 {
                        return Err(msg(format!("can't squeeze axis {x} of {:?}", t.dims())));
                    }
                    t = t.squeeze(x)?;
                }
                t
            }
        },
        Op::Slice { attrs } => {
            let (starts, ends, axes): (Vec<i64>, Vec<i64>, Vec<i64>) = match attrs {
                Some((s, e, ax)) => (s.clone(), e.clone(), ax.clone()),
                None => {
                    let (s, e) = (ints(a, 1)?.to_vec(), ints(a, 2)?.to_vec());
                    let ax = match arg(a, 3) {
                        Some(_) => ints(a, 3)?.to_vec(),
                        None => (0..s.len() as i64).collect(),
                    };
                    if arg(a, 4).is_some() && ints(a, 4)?.iter().any(|s| *s != 1) {
                        return Err(msg("only step 1 slices are supported"));
                    }
                    (s, e, ax)
                }
            };
            if starts.len() != ends.len() || starts.len() != axes.len() {
                return Err(msg("starts, ends and axes differ in length"));
            }
            match arg(a, 0) {
                Some(Value::I(v)) => {
                    let (&[s], &[e], &[ax]) = (&starts[..], &ends[..], &axes[..]) else {
                        return Err(msg("slicing a shape takes one axis"));
                    };
                    axis(ax, 1)?;
                    let (s, e) = slice_range(s, e, v.len());
                    return Ok(Value::I(v.get(s..e).unwrap_or_default().to_vec()));
                }
                _ => {
                    let mut t = tensor(a, 0)?.clone();
                    for ((s, e), ax) in starts.iter().zip(&ends).zip(&axes) {
                        let ax = axis(*ax, t.rank())?;
                        let (s, e) = slice_range(*s, *e, t.dim(ax)?);
                        t = t.narrow(ax, s, e - s)?;
                    }
                    t
                }
            }
        }
        Op::Concat { axis: ax } => {
            if a.iter().all(|v| matches!(v, Some(Value::I(_)))) {
                let mut out = Vec::new();
                for i in 0..a.len() {
                    out.extend_from_slice(ints(a, i)?);
                }
                return Ok(Value::I(out));
            }
            let ts = (0..a.len()).map(|i| tensor(a, i)).collect::<Result<Vec<_>>>()?;
            let rank = ts.first().map_or(0, |t| t.rank());
            Tensor::cat(&ts, axis(*ax, rank)?)?
        }
        Op::Resize => {
            let x = tensor(a, 0)?;
            let (n, c, h, w) = x.dims4()?;
            if h == 0 || w == 0 {
                return Err(msg("empty input"));
            }
            let sizes = match arg(a, 3) {
                Some(Value::I(v)) if !v.is_empty() => Some(v.as_slice()),
                Some(Value::F(t)) if t.elem_count() > 0 => return Err(msg("sizes must be int64")),
                _ => None,
            };
            let (oh, ow, sh, sw) = match sizes {
                Some(&[sn, sc, oh, ow]) => {
                    let (oh, ow) = (usize::try_from(oh).map_err(|_| msg("negative size"))?, usize::try_from(ow).map_err(|_| msg("negative size"))?);
                    if sn != n as i64 || sc != c as i64 || oh == 0 || ow == 0 {
                        return Err(msg(format!("only spatial resizing is supported (sizes {sizes:?})")));
                    }
                    (oh, ow, oh as f32 / h as f32, ow as f32 / w as f32)
                }
                Some(s) => return Err(msg(format!("sizes {s:?} must have 4 values"))),
                None => {
                    let s = tensor(a, 2)?.flatten_all()?.to_vec1::<f32>()?;
                    let &[s0, s1, sh, sw] = &s[..] else { return Err(msg("scales must have 4 values")) };
                    if s0 != 1.0 || s1 != 1.0 || !(sh > 0.0 && sw > 0.0 && sh.is_finite() && sw.is_finite()) {
                        return Err(msg(format!("unsupported scales {s:?}")));
                    }
                    let (oh, ow) = ((h as f32 * sh).floor(), (w as f32 * sw).floor());
                    if !(oh >= 1.0 && ow >= 1.0 && oh < 1e9 && ow < 1e9) {
                        return Err(msg(format!("unsupported scales {s:?}")));
                    }
                    (oh as usize, ow as usize, sh, sw)
                }
            };
            checked_elements(&[n, c, oh, ow])?;
            let index = |out: usize, scale: f32, len: usize| -> Result<Tensor> {
                let v: Vec<u32> = (0..out).map(|i| ((i as f32 / scale).floor().max(0.0) as usize).min(len - 1) as u32).collect();
                Ok(Tensor::from_vec(v, out, x.device())?)
            };
            x.index_select(&index(oh, sh, h)?, 2)?.index_select(&index(ow, sw, w)?, 3)?
        }
        Op::Transpose { perm } => {
            let t = tensor(a, 0)?;
            let perm = perm.clone().unwrap_or_else(|| (0..t.rank()).rev().collect());
            if perm.len() != t.rank() {
                return Err(msg(format!("perm {perm:?} does not match rank {}", t.rank())));
            }
            t.permute(perm)?
        }
        Op::Reshape => {
            let (t, shape) = (tensor(a, 0)?, ints(a, 1)?);
            let mut out = Vec::with_capacity(shape.len());
            let mut infer = None;
            for (i, s) in shape.iter().enumerate() {
                match *s {
                    0 => out.push(*t.dims().get(i).ok_or_else(|| msg("0 in a shape beyond the input's rank"))?),
                    -1 if infer.is_none() => {
                        infer = Some(i);
                        out.push(1);
                    }
                    s if s > 0 => out.push(usize::try_from(s).map_err(|_| msg("bad shape"))?),
                    _ => return Err(msg(format!("bad shape {shape:?}"))),
                }
            }
            let known = checked_elements(&out)?;
            if let Some(i) = infer {
                if known == 0 || t.elem_count() % known != 0 {
                    return Err(msg(format!("can't reshape {:?} to {shape:?}", t.dims())));
                }
                if let Some(d) = out.get_mut(i) {
                    *d = t.elem_count() / known;
                }
            }
            t.contiguous()?.reshape(out)?
        }
    };
    Ok(Value::F(out))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{Graph, Result};
    use crate::onnx::{self, encode::*};

    fn graph(bytes: &[u8]) -> Result<Graph> {
        Graph::new(onnx::parse(bytes)?, &Device::Cpu)
    }

    fn run1(g: &Graph, x: Tensor) -> Vec<f32> {
        g.run(&x).unwrap()[0].flatten_all().unwrap().to_vec1().unwrap()
    }

    #[test]
    fn same_upper_pads_at_the_end() {
        // a 2×2 max pool, stride 1, SAME_UPPER: the output keeps the input's size, the last row/column see the edge
        let m = model(
            &[node(
                "MaxPool",
                &["x"],
                &["y"],
                &[attr_ints("kernel_shape", &[2, 2]), attr_ints("strides", &[1, 1]), attr_str("auto_pad", "SAME_UPPER")],
            )],
            &[],
            &["x"],
            &["y"],
        );
        let g = graph(&m).unwrap();
        let x = Tensor::from_vec(vec![1f32, 5., 3., 2.], (1, 1, 2, 2), &Device::Cpu).unwrap();
        assert_eq!(run1(&g, x), vec![5., 5., 3., 2.]);
    }

    #[test]
    fn depthwise_taps_match_candles_grouped_convolution() {
        // 4 channels, 9×7 input: kernels 3×3 and 5×5, strides 1 and 2, padding, a dilation; against candle's own
        // per-group convolution
        let x: Vec<f32> = (0..4 * 9 * 7).map(|v| ((v * 37 % 23) as f32 - 11.0) / 7.0).collect();
        let x = Tensor::from_vec(x, (1, 4, 9, 7), &Device::Cpu).unwrap();
        for (k, stride, pad, dil) in [(3, 1, 1, 1), (3, 2, 1, 1), (5, 2, 2, 1), (5, 1, 2, 1), (3, 1, 2, 2)] {
            let w: Vec<f32> = (0..4 * k * k).map(|v| ((v * 13 % 17) as f32 - 8.0) / 9.0).collect();
            let w = Tensor::from_vec(w, (4, 1, k, k), &Device::Cpu).unwrap();
            let want = x.conv2d(&w, pad, stride, dil, 4).unwrap();
            let got = super::depthwise(&x, &w, pad, [stride, stride], dil).unwrap();
            assert_eq!(got.dims(), want.dims(), "k {k} stride {stride}");
            let diff = (got - want).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            assert!(diff < 1e-4, "k {k} stride {stride} pad {pad} dilation {dil}: {diff}");
        }
        // a window larger than the map is an error, not a panic
        let w = Tensor::ones((4, 1, 3, 3), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(super::depthwise(&x.narrow(2, 0, 2).unwrap(), &w, 0, [1, 1], 1).is_err());
    }

    #[test]
    fn pointwise_convolution_is_a_matrix_product() {
        // 3 → 5 channels, 1×1, with bias: against candle's conv2d
        let xs: Vec<f32> = (0..2 * 3 * 4 * 6).map(|v| ((v * 31 % 19) as f32 - 9.0) / 5.0).collect();
        let ws: Vec<f32> = (0..5 * 3).map(|v| ((v * 7 % 11) as f32 - 5.0) / 4.0).collect();
        let m = model(
            &[node("Conv", &["x", "w", "b"], &["y"], &[attr_ints("kernel_shape", &[1, 1])])],
            &[tensor_f32("w", &[5, 3, 1, 1], &ws), tensor_f32("b", &[5], &[0.5, -1.0, 0.0, 2.0, 0.25])],
            &["x"],
            &["y"],
        );
        let g = graph(&m).unwrap();
        let x = Tensor::from_vec(xs, (2, 3, 4, 6), &Device::Cpu).unwrap();
        let w = Tensor::from_vec(ws, (5, 3, 1, 1), &Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![0.5f32, -1.0, 0.0, 2.0, 0.25], (1, 5, 1, 1), &Device::Cpu).unwrap();
        let want: Vec<f32> = x.conv2d(&w, 0, 1, 1, 1).unwrap().broadcast_add(&b).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let got = run1(&g, x);
        assert_eq!(got.len(), want.len());
        assert!(got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 1e-4), "{got:?} vs {want:?}");
    }

    #[test]
    fn unequal_strides_subsample_rows_only() {
        // 1×1 conv (weight 1) with strides [2, 1] on a 4×2 input: rows 0 and 2, both columns
        let m = model(
            &[node("Conv", &["x", "w"], &["y"], &[attr_ints("kernel_shape", &[1, 1]), attr_ints("strides", &[2, 1])])],
            &[tensor_f32("w", &[1, 1, 1, 1], &[1.0])],
            &["x"],
            &["y"],
        );
        let g = graph(&m).unwrap();
        let x = Tensor::from_vec((0..8).map(|v| v as f32).collect::<Vec<_>>(), (1, 1, 4, 2), &Device::Cpu).unwrap();
        assert_eq!(run1(&g, x), vec![0., 1., 4., 5.]);
    }

    #[test]
    fn hard_sigmoid_erf_and_layer_norm_ops() {
        let m = model(
            &[
                node("HardSigmoid", &["x"], &["h"], &[]),
                node("ReduceMean", &["h"], &["m"], &[attr_ints("axes", &[-1])]),
                node("Sub", &["h", "m"], &["d"], &[]),
                node("Pow", &["d", "two"], &["p"], &[]),
                node("Erf", &["p"], &["e"], &[]),
                node("Sqrt", &["e"], &["s"], &[]),
                node("Div", &["s", "two"], &["y"], &[]),
            ],
            &[tensor_f32("two", &[], &[2.0])],
            &["x"],
            &["y"],
        );
        let g = graph(&m).unwrap();
        let x = Tensor::from_vec(vec![-10f32, 0., 10.], (1, 3), &Device::Cpu).unwrap();
        let y = run1(&g, x);
        // hard sigmoid (0.2x + 0.5, clamped): 0, 0.5, 1; mean 0.5; squared deviations 0.25, 0, 0.25
        let want: Vec<f32> = [0.25f32, 0.0, 0.25].iter().map(|v| libm_erf(*v).sqrt() / 2.0).collect();
        for (a, b) in y.iter().zip(&want) {
            assert!((a - b).abs() < 1e-5, "{y:?} vs {want:?}");
        }
    }

    /// erf by its series (enough for small arguments).
    fn libm_erf(x: f32) -> f32 {
        let x = f64::from(x);
        let mut sum = 0.0;
        let mut term = x;
        for n in 0..30 {
            sum += term / (2 * n + 1) as f64;
            term *= -x * x / (n + 1) as f64;
        }
        (sum * 2.0 / std::f64::consts::PI.sqrt()) as f32
    }

    #[test]
    fn softmax_before_opset_13_flattens_the_trailing_axes() {
        let m = model(&[node("Softmax", &["x"], &["y"], &[attr_int("axis", 1)])], &[], &["x"], &["y"]);
        let g = graph(&m).unwrap();
        let x = Tensor::zeros((1, 2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        // the four values share one softmax (opset 11): 0.25 each, not 0.5
        assert_eq!(run1(&g, x), vec![0.25; 4]);
    }

    /// A Conv node over `x` with weights of `shape` (ones) and the given attributes.
    fn conv(shape: &[i64], attrs: &[Vec<u8>]) -> Vec<u8> {
        let n = shape.iter().product::<i64>().max(0) as usize;
        model(&[node("Conv", &["x", "w"], &["y"], attrs)], &[tensor_f32("w", shape, &vec![1.0; n])], &["x"], &["y"])
    }

    /// A hostile model file: a zero-size kernel, a huge dilation or padding are refused at load, and an input smaller than
    /// the window is an error, never a panic or an overflow.
    #[test]
    fn hostile_convolution_shapes_are_errors() {
        let same = attr_str("auto_pad", "SAME_UPPER");
        for (shape, attrs) in [
            (vec![1, 1, 0, 3], vec![same.clone()]),
            (vec![1, 1, 3, 0], vec![]),
            (vec![1, 1, 5, 5], vec![same.clone(), attr_ints("dilations", &[i64::MAX / 2, i64::MAX / 2])]),
            (vec![1, 1, 3, 3], vec![attr_ints("dilations", &[1 << 20, 1 << 20])]),
            (vec![1, 1, 3, 3], vec![attr_ints("pads", &[1 << 40, 0, 0, 0])]),
        ] {
            assert!(graph(&conv(&shape, &attrs)).is_err(), "{shape:?}");
        }
        // a feature map smaller than the window, on the general path (and a strided one): an error at run time
        let x = Tensor::ones((1, 1, 2, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        for attrs in [vec![], vec![attr_ints("strides", &[2, 1])], vec![attr_ints("dilations", &[2, 2])]] {
            let g = graph(&conv(&[1, 1, 3, 3], &attrs)).unwrap();
            assert!(g.run(&x).is_err());
        }
        // padded up to the window it runs
        let g = graph(&conv(&[1, 1, 3, 3], &[attr_ints("pads", &[1, 1, 1, 1])])).unwrap();
        assert_eq!(run1(&g, x), vec![4.0; 4]);
    }

    /// A kernel too large to pad for is refused at load, not left to exhaust memory in a pad.
    #[test]
    fn a_huge_kernel_is_refused_at_load() {
        // pooling: the kernel comes straight from an attribute
        let pool = model(
            &[node("MaxPool", &["x"], &["y"], &[attr_ints("kernel_shape", &[1 << 40, 1]), attr_str("auto_pad", "SAME_UPPER")])],
            &[],
            &["x"],
            &["y"],
        );
        assert!(graph(&pool).is_err());
        // a convolution: the kernel comes from the weights
        assert!(graph(&conv(&[1, 1, 2000, 1], &[attr_str("auto_pad", "SAME_UPPER")])).is_err());
        // an ordinary one still loads
        assert!(graph(&conv(&[1, 1, 3, 3], &[attr_str("auto_pad", "SAME_UPPER")])).is_ok());
    }

    /// Shape's start / end (opset 15) would slice the dims; they are not implemented, so they fail at load.
    #[test]
    fn shape_with_start_or_end_is_refused_at_load() {
        for attr in [attr_int("start", 1), attr_int("end", 2)] {
            let m = model(&[node("Shape", &["x"], &["s"], &[attr])], &[], &["x"], &["s"]);
            assert!(graph(&m).is_err());
        }
        let m = model(&[node("Shape", &["x"], &["s"], &[])], &[], &["x"], &["s"]);
        assert!(graph(&m).is_ok());
    }

    #[test]
    fn unknown_operators_fail_at_load() {
        let m = model(&[node("NonMaxSuppression", &["x"], &["y"], &[])], &[], &["x"], &["y"]);
        assert!(graph(&m).err().unwrap().to_string().contains("not supported"));
        // a value used before it is computed
        let m = model(&[node("Relu", &["z"], &["y"], &[])], &[], &["x"], &["y"]);
        assert!(graph(&m).is_err());
    }
}
