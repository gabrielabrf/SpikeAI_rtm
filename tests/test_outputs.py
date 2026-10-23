import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from algorithm_rules import detect_events
from export_handler import draw_overlay, pipeline_status, save_data, save_video
from ground import refine_with_ground
from validator import validate_video


def make_video(path, frames=12, fps=30, size=(160, 120)):
    rng = np.random.default_rng(0)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for _ in range(frames):
            image = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class TemporaryTestCase(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.tmp = Path(self._directory.name)


class ValidatorTests(TemporaryTestCase):
    def test_missing_file_is_rejected(self):
        result = validate_video(self.tmp / "nao_existe.mp4")
        self.assertEqual(result["status"], "REJEITAR")
        self.assertIn("não encontrado", result["motivos_rejeicao"][0])

    def test_file_without_video_stream_is_rejected(self):
        path = self.tmp / "texto.mp4"
        path.write_text("não é um vídeo")
        self.assertEqual(validate_video(path)["status"], "REJEITAR")

    def test_low_fps_is_accepted_with_alert(self):
        make_video(self.tmp / "v.mp4", fps=30)
        result = validate_video(self.tmp / "v.mp4")
        self.assertEqual(result["status"], "ACEITAR COM ALERTA")
        self.assertEqual((result["fps"], result["resolucao"]), (30.0, "160x120"))
        self.assertEqual(result["motivos_rejeicao"], [])

    def test_recommended_fps_has_no_alert(self):
        make_video(self.tmp / "v.mp4", fps=60)
        result = validate_video(self.tmp / "v.mp4")
        self.assertEqual((result["status"], result["alertas"]), ("ACEITAR", []))


class ExportTests(TemporaryTestCase):
    def setUp(self):
        super().setUp()
        self.states = np.array([1] * 10 + [0] * 15 + [1] * 10, dtype=np.int8)
        self.times = np.arange(len(self.states)) / 30
        self.events, self.final_states = detect_events(self.states, self.times, source="footskate")

    def test_complete_result_is_approved(self):
        status, alerts = pipeline_status(self.events, ["aviso de vídeo"])
        self.assertEqual((status, alerts), ("APROVADO", ["aviso de vídeo"]))

    def test_inconclusive_event_requires_manual_review(self):
        events, _ = detect_events([1] * 10 + [0] * 15, np.arange(25) / 30)
        status, alerts = pipeline_status(events)
        self.assertEqual(status, "PENDENTE_REVISAO_MANUAL")
        self.assertEqual(len(alerts), 1)
        self.assertTrue(alerts[0].startswith("landing:"))

    def test_saved_files(self):
        prob = np.full((len(self.states), 4), .5, dtype=np.float32)
        data = save_data(Path("clipe.mp4"), self.tmp, 0, 30.0, self.times, prob, self.final_states,
                         self.states, self.events, .5, {"available": False, "reason": "teste"}, ["aviso"])
        saved = json.loads((self.tmp / "events.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, data)
        self.assertEqual((saved["video_id"], saved["pipeline_status"], saved["alertas"]), ("clipe", "APROVADO", ["aviso"]))
        self.assertEqual([e["event"] for e in saved["events"]], ["take_off", "landing"])
        self.assertEqual(set(saved["definitions"]), {"take_off", "landing"})
        with (self.tmp / "events.csv").open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f, delimiter=";"))
        self.assertEqual([r["frame"] for r in rows], ["9", "25"])
        with (self.tmp / "contacts.csv").open(encoding="utf-8-sig") as f:
            contacts = list(csv.DictReader(f, delimiter=";"))
        self.assertEqual(len(contacts), len(self.states))
        self.assertEqual(list(contacts[0])[-2:], ["state_evidence", "state"])
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["contacts.csv", "events.csv", "events.json"])

    def test_overlay_has_even_dimensions_and_marks_inconclusive(self):
        events, _ = detect_events([1] * 10 + [0] * 15, np.arange(25) / 30)
        pose = np.zeros((25, 3), dtype=np.float32)
        image = draw_overlay(np.zeros((101, 99, 3), dtype=np.uint8), pose, np.zeros(4), 0, 3, 0.1, events)
        self.assertEqual((image.shape[0] % 2, image.shape[1] % 2), (0, 0))

    @unittest.skipUnless("libx264" in av.codecs_available, "libx264 indisponível")
    def test_inspection_video_has_every_frame(self):
        make_video(self.tmp / "v.mp4", frames=12)
        poses = np.zeros((12, 25, 3), dtype=np.float32)
        events, states = detect_events([1] * 12, np.arange(12) / 30)
        save_video(self.tmp / "v.mp4", self.tmp / "inspection.mp4", poses, np.zeros((12, 4)), states,
                   np.arange(12) / 30, events)
        with av.open(str(self.tmp / "inspection.mp4")) as container:
            self.assertEqual(sum(1 for _ in container.decode(video=0)), 12)


class GroundFallbackTests(TemporaryTestCase):
    def test_without_contacts_there_is_no_reference_and_no_states(self):
        make_video(self.tmp / "v.mp4", frames=12)
        poses = np.zeros((12, 25, 3), dtype=np.float32)
        states, lines, info, regions = refine_with_ground(self.tmp / "v.mp4", poses, np.zeros((12, 4)))
        self.assertIsNone(states)
        self.assertIsNone(lines)
        self.assertIsNone(regions)
        self.assertFalse(info["available"])
        self.assertTrue(info["reason"])


if __name__ == "__main__":
    unittest.main()
