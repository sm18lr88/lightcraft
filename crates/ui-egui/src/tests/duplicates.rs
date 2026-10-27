//! Photo > Find Duplicates…: the dialog groups a library's exact copies (no model), ticks the suggested photo of each
//! group to keep, and its button marks the other copy Rejected without deleting anything.

use std::time::Duration;

use lightcraft_catalog::{Flag, Op};
use serde_json::json;

use crate::headless::Headless;
use crate::state::Dialog;
use crate::{LightcraftApp, Services};

const T: Duration = Duration::from_secs(20);

fn click(h: &mut Headless, id: &str) {
    let r = h.request("ui.clickWidget", json!({"id": id}), T);
    assert_eq!(r["ok"], true, "{id}: {r}");
    h.step();
}

#[test]
fn the_dialog_keeps_the_suggested_copy_and_rejects_the_other() {
    let mut session = lightcraft_engine::Session::with_demo();
    let original = session.catalog.photos().map(|p| p.id).min().unwrap();
    // the same demo picture again, at half the size: an exact copy by content, with fewer pixels
    let mut copy = (**session.catalog.photo(original).unwrap()).clone();
    copy.id = session.catalog.alloc_photo_id();
    copy.width /= 2;
    copy.height /= 2;
    let copy_id = copy.id;
    session.catalog.apply(Op::AddPhoto { photo: Box::new(copy) }).unwrap();
    let photos = session.catalog.photos().count();
    let original_flag = session.catalog.photo(original).unwrap().flag;
    let app = LightcraftApp::new(session, Services { png: None, ..Default::default() });
    let mut h = Headless::new(app, [1400.0, 1000.0], 1.0);
    h.step();

    h.app.run("dialog.duplicates", json!({})).unwrap();
    h.step();
    h.step();
    assert!(matches!(&h.app.ui.dialog, Some(Dialog::Duplicates { near: true, .. })));
    click(&mut h, "radio:duplicates.exact");
    click(&mut h, "button:findDuplicates");
    assert!(h.step_until(T, |h| matches!(&h.app.ui.dialog, Some(Dialog::Duplicates { result: Some(_), .. }))), "the scan finishes");
    let Some(Dialog::Duplicates { result: Some(found), keep, .. }) = &h.app.ui.dialog else { panic!("the dialog shows the groups") };
    let groups = found["groups"].as_array().unwrap();
    assert_eq!(groups.len(), 1, "{found}");
    assert_eq!(groups[0]["keep"], original.0, "the bigger copy is suggested");
    assert!(keep.contains(&original.0) && !keep.contains(&copy_id.0), "the suggested photo is ticked: {keep:?}");

    click(&mut h, "button:duplicates.resolve.0");
    assert_eq!(h.app.session.catalog.photo(copy_id).unwrap().flag, Flag::Reject);
    assert_eq!(h.app.session.catalog.photo(original).unwrap().flag, original_flag, "the kept photo is left as it was");
    assert_eq!(h.app.session.catalog.photos().count(), photos, "nothing is deleted");
    let Some(Dialog::Duplicates { result: Some(found), .. }) = &h.app.ui.dialog else { panic!("the dialog stays open") };
    assert_eq!(found["groups"], json!([]), "the decided group leaves the list");
    click(&mut h, "button:dialogOk");
    assert!(h.app.ui.dialog.is_none(), "the dialog only shows things: its one button closes it");
}

/// Unticking every photo of a group leaves its button disabled: nothing is rejected without a photo kept.
#[test]
fn a_group_with_nothing_ticked_cannot_be_rejected() {
    let mut session = lightcraft_engine::Session::with_demo();
    let original = session.catalog.photos().map(|p| p.id).min().unwrap();
    let mut copy = (**session.catalog.photo(original).unwrap()).clone();
    copy.id = session.catalog.alloc_photo_id();
    let copy_id = copy.id;
    session.catalog.apply(Op::AddPhoto { photo: Box::new(copy) }).unwrap();
    let flags = (session.catalog.photo(original).unwrap().flag, session.catalog.photo(copy_id).unwrap().flag);
    let app = LightcraftApp::new(session, Services { png: None, ..Default::default() });
    let mut h = Headless::new(app, [1400.0, 1000.0], 1.0);
    h.step();
    h.app.run("dialog.duplicates", json!({})).unwrap();
    h.step();
    h.step();
    click(&mut h, "radio:duplicates.exact");
    click(&mut h, "button:findDuplicates");
    assert!(h.step_until(T, |h| matches!(&h.app.ui.dialog, Some(Dialog::Duplicates { result: Some(_), .. }))), "the scan finishes");
    if let Some(Dialog::Duplicates { keep, .. }) = &mut h.app.ui.dialog {
        keep.clear();
    }
    h.step();
    h.app.ui.toast = None;
    let r = h.request("ui.clickWidget", json!({"id": "button:duplicates.resolve.0"}), T);
    h.step();
    assert_eq!(r["ok"], true, "{r}");
    assert!(h.app.ui.toast.is_none(), "the disabled button must not invoke resolveDuplicates: {:?}", h.app.ui.toast);
    let now = (h.app.session.catalog.photo(original).unwrap().flag, h.app.session.catalog.photo(copy_id).unwrap().flag);
    assert_eq!(now, flags, "a disabled button rejects nothing");
}
