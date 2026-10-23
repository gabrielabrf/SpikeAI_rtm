import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from algorithm_rules import detect_events
from attacker import AttackerTracker
from footskate import contact_states, temporal_inputs, to_body25
from ground import estimate_ground, ground_states


class EventTests(unittest.TestCase):
    def events(self, states):
        return detect_events(states, np.arange(len(states)) / 30)[0]

    def test_last_support_and_first_return(self):
        events = self.events([1] * 10 + [0] * 15 + [1] * 10)
        self.assertEqual([e["frame"] for e in events], [9, 25])
        self.assertEqual([e["time_s"] for e in events], [0.3, 0.833333])

    def test_unknown_before_flight_does_not_invent_takeoff(self):
        events = self.events([1] * 10 + [-1] * 10 + [0] * 15 + [1] * 10)
        self.assertIsNone(events[0]["frame"])
        self.assertEqual(events[1]["frame"], 35)

    def test_unknown_after_flight_does_not_invent_landing(self):
        events = self.events([1] * 10 + [0] * 15 + [-1] * 10 + [1] * 10)
        self.assertEqual(events[0]["frame"], 9)
        self.assertIsNone(events[1]["frame"])

    def test_clip_starting_in_air(self):
        events = self.events([0] * 15 + [1] * 10)
        self.assertIsNone(events[0]["frame"])
        self.assertEqual(events[1]["frame"], 15)

    def test_clip_ending_in_air(self):
        events = self.events([1] * 10 + [0] * 15)
        self.assertEqual(events[0]["frame"], 9)
        self.assertIsNone(events[1]["frame"])

    def test_two_short_air_frames_do_not_confirm_a_jump(self):
        self.assertTrue(all(e["frame"] is None for e in self.events([1] * 10 + [0] * 2 + [1] * 10)))

    def test_unknown_is_not_air(self):
        self.assertTrue(all(e["frame"] is None for e in self.events([1] * 10 + [-1] * 20 + [1] * 10)))

    def test_two_comparable_flights_are_inconclusive(self):
        events = self.events([1] * 10 + [0] * 15 + [1] * 10 + [0] * 15 + [1] * 10)
        self.assertTrue(all(e["status"] == "inconclusive" for e in events))

    def test_timestamps_are_used_instead_of_frame_over_fps(self):
        states = [1] * 3 + [0] * 5 + [1] * 3
        times = np.array([0, .04, .08, .12, .15, .19, .23, .27, .31, .35, .39])
        events, _ = detect_events(states, times)
        self.assertEqual([e["time_s"] for e in events], [.08, .31])

    def test_non_increasing_times_rejected(self):
        with self.assertRaises(ValueError):
            detect_events([1, 0, 1], [0, 0, 1])

    def test_one_missing_frame_does_not_split_a_single_flight(self):
        events = self.events([1] * 10 + [0] * 8 + [-1] + [0] * 8 + [1] * 10)
        self.assertEqual([e["frame"] for e in events], [9, 27])

    def test_short_boundary_gap_reports_a_range(self):
        events = self.events([1] * 10 + [-1] * 2 + [0] * 15 + [1] * 10)
        self.assertEqual(events[0]["status"], "estimated")
        self.assertEqual(events[0]["frame_range"], [9, 11])
        self.assertIn(events[0]["frame"], [9, 10, 11])

    def test_long_boundary_gap_keeps_range_but_no_exact_frame(self):
        events = self.events([1] * 10 + [-1] * 10 + [0] * 15 + [1] * 10)
        self.assertEqual(events[0]["status"], "inconclusive")
        self.assertEqual(events[0]["frame_range"], [9, 19])
        self.assertIsNone(events[0]["frame"])

    def test_one_false_contact_frame_inside_flight_is_not_a_landing(self):
        events = self.events([1] * 10 + [0] * 8 + [1] + [0] * 8 + [1] * 10)
        self.assertEqual([e["frame"] for e in events], [9, 27])

    def test_geometry_result_is_explicitly_an_estimate(self):
        events, _ = detect_events([1] * 10 + [0] * 15 + [1] * 10,
                                  np.arange(35) / 30, source="footskate+ground_reference")
        self.assertTrue(all(e["status"] == "estimated" for e in events))


class PoseAndIdentityTests(unittest.TestCase):
    def test_halpe26_feet_mapping(self):
        pose = to_body25(np.arange(52).reshape(26, 2), np.ones(26))
        for body25, halpe in [(8, 19), (19, 20), (20, 22), (21, 24), (22, 21), (23, 23), (24, 25)]:
            self.assertEqual(pose[body25, 0], halpe * 2)

    def test_coco17_cannot_supply_feet(self):
        with self.assertRaises(ValueError):
            to_body25(np.zeros((17, 2)), np.ones(17))

    def test_one_foot_in_contact_keeps_support(self):
        poses = np.ones((10, 25, 3), dtype=np.float32)
        prob = np.tile([.9, .1, .1, .1], (10, 1))
        np.testing.assert_array_equal(contact_states(prob, poses, np.ones(10, bool)), 1)

    def test_missing_foot_is_not_proof_of_flight(self):
        poses = np.ones((10, 25, 3), dtype=np.float32)
        poses[:, 24, 2] = 0
        states = contact_states(np.full((10, 4), .1), poses, np.ones(10, bool))
        np.testing.assert_array_equal(states, -1)

    def test_no_pose_gives_invalid_network_input(self):
        inputs, valid = temporal_inputs(np.zeros((8, 25, 3), dtype=np.float32))
        self.assertFalse(valid.any())
        self.assertTrue(np.isfinite(inputs).all())

    def test_detection_order_does_not_change_attacker(self):
        tracker = AttackerTracker([100, 100, 150, 250])
        selected = tracker.update(np.array([[500, 100, 550, 250], [105, 100, 155, 250]]))
        self.assertEqual(selected, 1)

    def test_ambiguous_association_is_unknown(self):
        tracker = AttackerTracker([100, 100, 150, 250])
        self.assertIsNone(tracker.update(np.array([[98, 100, 148, 250], [102, 100, 152, 250]])))

    def test_lost_track_does_not_switch_to_other_person(self):
        tracker = AttackerTracker([100, 100, 150, 250], max_missing=2)
        tracker.update(np.empty((0, 4)))
        tracker.update(np.empty((0, 4)))
        self.assertIsNone(tracker.update(np.array([[100, 100, 150, 250]])))


class GroundTests(unittest.TestCase):
    def example(self):
        poses = np.ones((60, 25, 3), dtype=np.float32)
        poses[:, :, :2] = [50, 60]
        for hip, knee, ankle, x in [(9, 10, 11, 40), (12, 13, 14, 60)]:
            poses[:, hip, :2] = [x, 60]
            poses[:, knee, :2] = [x, 80]
            poses[:, ankle, :2] = [x, 100]
        for joint, x in [(19, 42), (21, 38), (22, 62), (24, 58)]:
            poses[:, joint, :2] = [x, 100]
        poses[15:45, :, 1] -= 25
        return poses, np.full((60, 4), .95), np.ones(60, dtype=bool)

    def test_model_support_in_air_is_rejected_with_ground_evidence(self):
        poses, prob, camera_ok = self.example()
        reference, info = estimate_ground(poses[:, :, :2], poses, prob, camera_ok)
        self.assertTrue(info["available"])
        states, _ = ground_states(poses[:, :, :2], poses, prob, camera_ok, reference, np.arange(60) / 30)
        self.assertEqual(states[5], 1)
        self.assertEqual(states[30], 0)
        self.assertEqual(states[55], 1)

    def test_no_contacts_cannot_calibrate_floor(self):
        poses, prob, camera_ok = self.example()
        reference, info = estimate_ground(poses[:, :, :2], poses, prob * 0, camera_ok)
        self.assertIsNone(reference)
        self.assertFalse(info["available"])

    def test_unreliable_camera_does_not_create_floor(self):
        poses, prob, camera_ok = self.example()
        camera_ok[20:] = False
        reference, info = estimate_ground(poses[:, :, :2], poses, prob, camera_ok)
        self.assertIsNone(reference)

    def test_occluded_foot_does_not_confirm_air(self):
        poses, prob, camera_ok = self.example()
        reference, _ = estimate_ground(poses[:, :, :2], poses, prob, camera_ok)
        poses[30, 24, 2] = 0
        states, _ = ground_states(poses[:, :, :2], poses, prob, camera_ok, reference, np.arange(60) / 30)
        self.assertEqual(states[30], -1)


if __name__ == "__main__":
    unittest.main()
