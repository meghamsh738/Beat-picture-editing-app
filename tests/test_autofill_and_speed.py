import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

from PySide6 import QtCore, QtWidgets

import os

import timeline_builder as tb


class TestAutofillLimit(unittest.TestCase):
    def test_compute_marker_time_with_head(self):
        marker_frames = [0, 30, 50]
        fps = 10
        head_frames = 20
        self.assertAlmostEqual(tb.compute_marker_time(marker_frames, fps, head_frames, 2), 5.0)

    def test_segments_within_autofill_limit(self):
        seg0 = tb.Segment(name="Intro", duration=2.0, kind="black")
        seg0.start = 0.0
        seg0.end = 2.0
        seg1 = tb.Segment(name="S1", duration=3.0, kind="image")
        seg1.start = 2.0
        seg1.end = 5.0
        seg2 = tb.Segment(name="S2", duration=2.0, kind="image")
        seg2.start = 5.0
        seg2.end = 7.0
        limit_time = 5.0
        indices = tb.segments_within_autofill_limit([seg0, seg1, seg2], limit_time)
        self.assertEqual(indices, [1])
        self.assertEqual(tb.segments_within_autofill_limit([seg0, seg1, seg2], None), [1, 2])


class TestRenderSpeed(unittest.TestCase):
    def test_resolve_render_speed_fast(self):
        settings = tb.resolve_render_speed("fast", cpu_count=8)
        self.assertEqual(settings["x264_preset"], "veryfast")
        self.assertEqual(settings["nvenc_preset"], "p2")
        self.assertEqual(settings["threads"], 8)

    def test_resolve_render_speed_fallback(self):
        settings = tb.resolve_render_speed("unknown", cpu_count=12)
        self.assertEqual(settings["x264_preset"], "faster")
        self.assertEqual(settings["nvenc_preset"], "p4")
        self.assertEqual(settings["threads"], 4)


class TestAutofillImagePool(unittest.TestCase):
    def test_resolve_autofill_images(self):
        all_images = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
        selected = ["b.jpg", "c.jpg"]
        pool = tb.resolve_autofill_images(selected, all_images)
        self.assertEqual(pool, [Path("b.jpg"), Path("c.jpg")])

    def test_resolve_autofill_images_fallback(self):
        all_images = [Path("a.jpg"), Path("b.jpg")]
        selected = ["missing.jpg"]
        pool = tb.resolve_autofill_images(selected, all_images)
        self.assertEqual(pool, all_images)


class TestUnassignImages(unittest.TestCase):
    def test_unassign_image_from_segments(self):
        normalizer = lambda p: os.path.normcase(os.path.normpath(p))
        seg_a = tb.Segment(name="S1", duration=1.0, kind="image", image_path="/tmp/a.jpg")
        seg_b = tb.Segment(name="S2", duration=1.0, kind="image", image_path="/tmp/b.jpg")
        seg_black = tb.Segment(name="S3", duration=1.0, kind="black", image_path="/tmp/a.jpg")
        segments = [seg_a, seg_b, seg_black]
        removed = tb.unassign_image_from_segments(segments, "/tmp/a.jpg", normalizer)
        self.assertEqual(removed, 1)
        self.assertEqual(seg_a.image_path, "")
        self.assertEqual(seg_b.image_path, "/tmp/b.jpg")
        self.assertEqual(seg_black.image_path, "/tmp/a.jpg")


class TestHistoryStack(unittest.TestCase):
    def test_history_undo_redo(self):
        stack = tb.HistoryStack()
        stack.reset({"segments": [1]})
        self.assertFalse(stack.can_undo)
        stack.push({"segments": [1, 2]})
        self.assertTrue(stack.can_undo)
        state = stack.undo()
        self.assertEqual(state, {"segments": [1]})
        self.assertTrue(stack.can_redo)
        state = stack.redo()
        self.assertEqual(state, {"segments": [1, 2]})


class TestBundleStartIndex(unittest.TestCase):
    def test_bundle_start_prefers_after_last_assigned(self):
        segs = [
            tb.Segment(name="S1", duration=1.0, kind="image", image_path="a.jpg"),
            tb.Segment(name="S2", duration=1.0, kind="image", image_path="b.jpg"),
            tb.Segment(name="S3", duration=1.0, kind="image", image_path=""),
        ]
        start, fallback = tb.compute_bundle_start_index(segs)
        self.assertEqual(start, 2)
        self.assertFalse(fallback)

    def test_bundle_start_falls_back_to_first_empty(self):
        segs = [
            tb.Segment(name="S1", duration=1.0, kind="image", image_path=""),
            tb.Segment(name="S2", duration=1.0, kind="image", image_path="b.jpg"),
        ]
        start, fallback = tb.compute_bundle_start_index(segs)
        self.assertEqual(start, 0)
        self.assertTrue(fallback)

    def test_bundle_start_none_when_no_empty(self):
        segs = [
            tb.Segment(name="S1", duration=1.0, kind="image", image_path="a.jpg"),
            tb.Segment(name="S2", duration=1.0, kind="black", image_path=""),
        ]
        start, fallback = tb.compute_bundle_start_index(segs)
        self.assertIsNone(start)
        self.assertFalse(fallback)


class TestResizableDialogs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_image_preview_dialog_is_resizable(self):
        dialog = tb.ImagePreviewDialog("missing-image.jpg")
        self.assertTrue(dialog.isSizeGripEnabled())
        flags = dialog.windowFlags()
        self.assertTrue(flags & QtCore.Qt.WindowMaximizeButtonHint)


class TestLibraryUsageHighlights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_library_usage_flags(self):
        widget = tb.ImageLibraryWidget()
        widget.set_images({"folder": [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]})
        while widget._pending_batches:
            widget._process_thumbnail_queue()
        normalizer = lambda p: os.path.normcase(os.path.normpath(p))
        used = {normalizer("/tmp/b.jpg")}
        widget.apply_usage_highlights(used, normalizer)
        item_a = widget._path_map["/tmp/a.jpg"][1]
        item_b = widget._path_map["/tmp/b.jpg"][1]
        self.assertFalse(item_a.data(tb.LIB_USED_FLAG_ROLE))
        self.assertTrue(item_b.data(tb.LIB_USED_FLAG_ROLE))
        widget._loader_timer.stop()
        widget.deleteLater()
        QtWidgets.QApplication.processEvents()

    def test_library_usage_pending_applies_after_load(self):
        widget = tb.ImageLibraryWidget()
        widget.set_images({"folder": [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")]})
        normalizer = lambda p: os.path.normcase(os.path.normpath(p))
        used = {normalizer("/tmp/a.jpg")}
        widget.apply_usage_highlights(used, normalizer)
        while widget._pending_batches:
            widget._process_thumbnail_queue()
        item_a = widget._path_map["/tmp/a.jpg"][1]
        self.assertTrue(item_a.data(tb.LIB_USED_FLAG_ROLE))
        widget._loader_timer.stop()
        widget.deleteLater()
        QtWidgets.QApplication.processEvents()


class TestAudioOffsetArgs(unittest.TestCase):
    def test_audio_offset_zero(self):
        self.assertEqual(tb._build_audio_input_args(0.0, "audio.wav"), ["-i", "audio.wav"])

    def test_audio_offset_positive(self):
        args = tb._build_audio_input_args(1.25, "audio.wav")
        self.assertEqual(args[:2], ["-itsoffset", "1.250"])
        self.assertEqual(args[-2:], ["-i", "audio.wav"])

    def test_audio_offset_negative(self):
        args = tb._build_audio_input_args(-0.5, "audio.wav")
        self.assertEqual(args[:2], ["-ss", "0.500"])
        self.assertEqual(args[-2:], ["-i", "audio.wav"])


if __name__ == "__main__":
    unittest.main()
