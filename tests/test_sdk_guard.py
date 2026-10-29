from __future__ import annotations
import json
import threading
import time
import unittest
import numpy as np
from poise.deploy.sharpa_guarded import GUARDED_HELPER_PROGRAM, GuardedSharpaControlClient, GuardedControlStatus

class GuardedSharpaBridgeTest(unittest.TestCase):
    def test_absolute_guard_caps_are_enforced_by_host(self) -> None:
        with self.assertRaisesRegex(ValueError, "0.015"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                max_step_rad=0.0151,
            )
        with self.assertRaisesRegex(ValueError, "0.010"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                policy_pulse_max_step_rad=0.0101,
            )
        with self.assertRaisesRegex(ValueError, "0.010"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                policy_burst_max_step_rad=0.0101,
            )
        with self.assertRaisesRegex(ValueError, "0.025"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                policy_burst_cumulative_limit_rad=0.0251,
            )
        with self.assertRaisesRegex(ValueError, "5, 10, or 20"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                policy_burst_max_commands=6,
            )
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                max_step_rad=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                heartbeat_timeout_s=float("inf"),
            )
        with self.assertRaisesRegex(ValueError, "0.30"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                max_speed_coeff=0.31,
            )
        with self.assertRaisesRegex(ValueError, "0.30"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                max_current_coeff=0.31,
            )
        with self.assertRaisesRegex(ValueError, "0.50"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                heartbeat_timeout_s=0.51,
            )
        with self.assertRaisesRegex(ValueError, "hard limits"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                session_target_sdk_rad=np.full(22, 3.0),
            )
        with self.assertRaisesRegex(ValueError, "measured.*hard limits"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                session_target_sdk_rad=np.zeros(22),
                session_measured_target_sdk_rad=np.full(22, 3.0),
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 0\.020\]"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                close_corridor_tolerance_rad=0.0201,
            )
        wedge50 = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
            session_target_sdk_rad=np.full(22, -0.30756178),
            session_measured_target_sdk_rad=np.full(22, -0.33342233),
            close_corridor_tolerance_rad=0.020,
        )
        self.assertLessEqual(
            -0.34480,
            float(wedge50.session_measured_target_sdk_rad[1])
            + wedge50.close_corridor_tolerance_rad,
        )
        self.assertGreaterEqual(
            -0.34480,
            float(wedge50.session_measured_target_sdk_rad[1])
            - wedge50.close_corridor_tolerance_rad,
        )
        with self.assertRaisesRegex(ValueError, "explicit 22D policy limits"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                exact_policy_enabled=True,
            )
        with self.assertRaisesRegex(ValueError, "inside guarded hard limits"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                exact_policy_enabled=True,
                exact_policy_lower_sdk_rad=np.full(22, -1.1),
                exact_policy_upper_sdk_rad=np.full(22, 1.0),
            )
        arbitrary = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
            exact_policy_max_commands=30,
        )
        self.assertEqual(arbitrary.exact_policy_max_commands, 30)
        with self.assertRaisesRegex(ValueError, "positive"):
            GuardedSharpaControlClient(
                np.full(22, -1.0),
                np.full(22, 2.0),
                exact_policy_max_commands=0,
            )

    def test_guarded_state_protocol_validation(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )
        client._handle_protocol_record(
            {
                "kind": "ready",
                "serial": "test",
                "hand_side": "RIGHT",
                "control_mode": "POSITION",
                "control_source": "SDK",
                "enable_state": True,
                "speed_coeff": 0.12,
                "current_coeff": 0.30,
                "max_step_rad": 0.015,
                "policy_pulse_max_step_rad": 0.010,
                "policy_burst_max_step_rad": 0.010,
                "policy_burst_cumulative_limit_rad": 0.025,
                "policy_burst_max_commands": 5,
                "pregrasp_fraction": 0.70,
                "heartbeat_timeout_s": 0.35,
            }
        )
        client._handle_protocol_record(
            {
                "kind": "state",
                "sequence": 10,
                "device_timestamp": 20,
                "angles_rad": [0.1] * 22,
                "velocities_rad_s": [0.0] * 22,
                "torques_nm": [0.02] * 22,
                "armed": True,
                "estop_latched": False,
                "commands_accepted": 7,
                "motion_phase": "close",
            }
        )
        state = client.latest(max_age_s=1.0)
        self.assertEqual(state.sequence, 10)
        np.testing.assert_allclose(state.torques_nm, np.full(22, 0.02))
        self.assertTrue(client.status.armed)
        self.assertEqual(client.status.commands_accepted, 7)
        self.assertEqual(client.status.motion_phase, "close")

    def test_command_rejection_is_not_overwritten_by_following_estop(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )
        client._handle_protocol_record(
            {
                "kind": "command_error",
                "command_id": "same-id",
                "operation": "target",
                "message": "step too large",
            }
        )
        client._handle_protocol_record(
            {
                "kind": "estop",
                "command_id": "same-id",
                "reason": "command_failure:target",
                "hardware_disable_succeeded": True,
            }
        )
        self.assertEqual(
            client._responses["same-id"]["kind"],
            "command_error",
        )
        self.assertTrue(client.status.estop_latched)

    def test_hold_response_returns_exact_helper_target(self) -> None:
        expected = np.linspace(-0.1, 0.2, 22)
        actual = GuardedSharpaControlClient._held_angles_from_response(
            {"held_angles_rad": expected.tolist()},
            "hold",
        )
        np.testing.assert_allclose(actual, expected)

    def test_target_requires_phase_token_for_helper_owned_limits(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_send(kind: str, **payload: object) -> dict[str, object]:
            calls.append((kind, payload))
            return {
                "kind": "ack",
                "operation": kind,
                "accepted_angles_sdk_rad": payload["angles_sdk_rad"],
            }

        client._send = fake_send  # type: ignore[method-assign]
        accepted = client.command_target(
            np.zeros(22),
            motion_phase="close",
        )
        np.testing.assert_array_equal(accepted, np.zeros(22))
        self.assertEqual(calls[0][0], "target")
        self.assertEqual(
            calls[0][1]["motion_phase"],
            "close",
        )
        self.assertNotIn("max_command_lead_rad", calls[0][1])
        with self.assertRaisesRegex(ValueError, "motion_phase"):
            client.command_target(
                np.zeros(22),
                motion_phase="unknown",
            )

    def test_pregrasp_target_projects_tiny_endpoint_skew_inward(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )
        held = np.full(22, 0.02026)
        endpoint = np.full(22, 0.21408)
        client._armed_hold_sdk_rad = held.copy()
        client._armed_pregrasp_target_sdk_rad = endpoint.copy()
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_send(kind: str, **payload: object) -> dict[str, object]:
            calls.append((kind, payload))
            return {
                "kind": "ack",
                "operation": kind,
                "accepted_angles_sdk_rad": payload["angles_sdk_rad"],
            }

        client._send = fake_send  # type: ignore[method-assign]
        requested = endpoint.copy()
        requested[17] += 0.00012
        accepted = client.command_target(
            requested,
            motion_phase="pregrasp",
        )

        self.assertEqual(float(accepted[17]), float(endpoint[17]))
        sent = np.asarray(calls[0][1]["angles_sdk_rad"], dtype=np.float64)
        self.assertEqual(float(sent[17]), float(endpoint[17]))

    def test_pregrasp_target_rejects_large_endpoint_disagreement(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )
        client._armed_hold_sdk_rad = np.zeros(22)
        client._armed_pregrasp_target_sdk_rad = np.ones(22)
        requested = np.ones(22)
        requested[17] += 0.001

        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            client.command_target(
                requested,
                motion_phase="pregrasp",
            )

    def test_pregrasp_hold_advances_helper_owned_resume_corridor(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )
        endpoint = np.full(22, 0.21408)
        client._armed_hold_sdk_rad = np.full(22, 0.02026)
        client._armed_pregrasp_target_sdk_rad = endpoint.copy()
        held = endpoint.copy()
        held[17] += 0.000897
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_send(kind: str, **payload: object) -> dict[str, object]:
            calls.append((kind, payload))
            if kind == "hold":
                return {
                    "kind": "ack",
                    "operation": kind,
                    "held_angles_rad": held.tolist(),
                    "pregrasp_resume_start_rad": held.tolist(),
                }
            return {
                "kind": "ack",
                "operation": kind,
                "accepted_angles_sdk_rad": payload["angles_sdk_rad"],
            }

        client._send = fake_send  # type: ignore[method-assign]
        np.testing.assert_array_equal(client.hold(), held)

        # This is the exact failure mode from the repeated-P incident: the
        # measured HOLD is just beyond the old arm-time endpoint.  Resuming
        # back toward the endpoint must remain bounded and must not reopen the
        # corridor toward the original arm pose.
        resumed = held.copy()
        resumed[17] -= 0.0005
        accepted = client.command_target(
            resumed,
            motion_phase="pregrasp",
        )
        np.testing.assert_array_equal(accepted, resumed)
        sent = np.asarray(calls[-1][1]["angles_sdk_rad"], dtype=np.float64)
        np.testing.assert_array_equal(sent, resumed)

    def test_pregrasp_hold_rejects_mismatched_resume_anchor(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )
        held = np.full(22, 0.2)
        resume = held.copy()
        resume[17] += 0.001

        def fake_send(kind: str, **_: object) -> dict[str, object]:
            self.assertEqual(kind, "hold")
            return {
                "kind": "ack",
                "operation": kind,
                "held_angles_rad": held.tolist(),
                "pregrasp_resume_start_rad": resume.tolist(),
            }

        client._send = fake_send  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "differs"):
            client.hold()

    def test_target_refresh_accepts_no_host_angles(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )
        calls: list[tuple[str, dict[str, object]]] = []

        def fake_send(kind: str, **payload: object) -> dict[str, object]:
            calls.append((kind, payload))
            return {"kind": "ack", "operation": kind}

        client._send = fake_send  # type: ignore[method-assign]
        client.refresh_target()
        self.assertEqual(
            calls,
            [
                (
                    "refresh_target",
                    {"wait_for_ack": True, "timeout_s": 0.5},
                )
            ],
        )

    def test_exact_client_accepts_extended_confirmation_and_last_index(
        self,
    ) -> None:
        for count in (37, 40, 100, 400, 800, 1200):
            with self.subTest(count=count):
                client = GuardedSharpaControlClient(
                    np.full(22, -1.0),
                    np.full(22, 2.0),
                    exact_policy_enabled=True,
                    exact_policy_max_commands=count,
                    exact_policy_lower_sdk_rad=np.full(22, -0.5),
                    exact_policy_upper_sdk_rad=np.full(22, 1.5),
                )
                calls: list[tuple[str, dict[str, object]]] = []

                def fake_send(
                    kind: str,
                    **payload: object,
                ) -> dict[str, object]:
                    calls.append((kind, payload))
                    return {
                        "kind": "ack",
                        "operation": kind,
                        "commands_accepted": 88,
                        "exact_policy_command_count": count,
                        "exact_policy_completed": True,
                    }

                client._send = fake_send  # type: ignore[method-assign]
                accepted = client.command_exact_policy(
                    np.zeros(22),
                    exact_index=count - 1,
                    confirmation="EXACT_POLICY_CONTRACT",
                )
                self.assertEqual(accepted, 88)
                self.assertEqual(calls[0][0], "exact_policy")
                self.assertEqual(calls[0][1]["exact_index"], count - 1)

    def test_arm_caches_helper_owned_pregrasp_target(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )
        held = np.linspace(-0.1, 0.1, 22)
        pregrasp = np.linspace(0.2, 0.4, 22)

        def fake_send(kind: str, **_: object) -> dict[str, object]:
            self.assertEqual(kind, "arm")
            return {
                "kind": "ack",
                "operation": "arm",
                "held_angles_rad": held.tolist(),
                "pregrasp_target_rad": pregrasp.tolist(),
            }

        client._send = fake_send  # type: ignore[method-assign]
        actual = client.arm(
            speed_coeff=0.12,
            current_coeff=0.30,
            confirmation="POISE",
        )

        np.testing.assert_array_equal(actual, held)
        np.testing.assert_array_equal(
            client.armed_pregrasp_target_sdk_rad,
            pregrasp,
        )
        copied = client.armed_pregrasp_target_sdk_rad
        copied[0] = 999.0
        self.assertNotEqual(
            client.armed_pregrasp_target_sdk_rad[0],
            copied[0],
        )

    def test_arm_timeout_requests_estop_before_raising(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )
        calls: list[str] = []

        def fake_send(kind: str, **_: object) -> dict[str, object]:
            calls.append(kind)
            if kind == "arm":
                raise TimeoutError("simulated arm timeout")
            return {
                "kind": "estop",
                "hardware_disable_succeeded": True,
            }

        client._send = fake_send  # type: ignore[method-assign]
        with self.assertRaisesRegex(TimeoutError, "simulated"):
            client.arm(
                speed_coeff=0.12,
                current_coeff=0.30,
                confirmation="POISE",
            )
        self.assertEqual(calls, ["arm", "estop"])

    def test_normal_stop_updates_cached_armed_status(self) -> None:
        client = GuardedSharpaControlClient(
            np.full(22, -1.0),
            np.full(22, 2.0),
        )

        class _RunningProcess:
            @staticmethod
            def poll() -> None:
                return None

        calls: list[str] = []
        client._process = _RunningProcess()  # type: ignore[assignment]
        client._status = GuardedControlStatus(True, False, 7)
        client._send = (  # type: ignore[method-assign]
            lambda kind, **_: calls.append(kind) or {"kind": "ack"}
        )
        client.normal_stop()
        self.assertEqual(calls, ["normal_stop"])
        self.assertFalse(client.status.armed)
        self.assertEqual(client.status.commands_accepted, 7)
