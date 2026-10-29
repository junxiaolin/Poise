"""RGB-D closed-loop control for a Sharpa Wave right hand.

Run without --execute to inspect tracking and joint state. Hardware execution
requires explicit --execute and keyboard arming in the preview window.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from poise.deploy.config import prepare_deployment
from poise.deploy.lifecycle import deployment_resources
from poise.deploy.control import PolicyController, transform_to_pose
from poise.deploy.hand_camera import AprilTagHandCameraEstimator, HandCameraDriftWatchdog, aggregate_hand_camera_estimates
from poise.deploy.realsense_runtime import RealSenseCaptureWorker, RealSenseSession, RealSenseStreamConfig
from poise.deploy.sharpa_guarded import GuardedSharpaControlClient
from poise.deploy.tracker import FoundationPoseObjectTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Camera, mesh, tag mounting and target configuration JSON")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("cycles must be positive")
    prepared = prepare_deployment(args.config, args.bundle)
    cfg, bundle, initial = prepared.config, prepared.bundle, prepared.initial
    q_goal, q_measured = initial["q_cmd"].numpy(), initial["q"].numpy()
    lower = np.asarray(bundle.manifest["sim_contract"]["joint_lower"])
    upper = np.asarray(bundle.manifest["sim_contract"]["joint_upper"])
    target_pose, mask = prepared.target_pose, prepared.mask
    gravity, mesh_to_object = cfg.gravity_direction_palm, cfg.t_mesh_object
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "preflight.json").write_text(json.dumps(prepared.report, indent=2))
    tracker = FoundationPoseObjectTracker(foundationpose_root=cfg.foundationpose_root, mesh_file=cfg.mesh,
                                          debug_dir=args.output / "tracking", t_mesh_object_sim=mesh_to_object)
    estimator = AprilTagHandCameraEstimator(tag_id=cfg.tag_id, tag_size_m=cfg.tag_size_m,
                                           t_hand_tag=cfg.t_hand_tag)
    client = GuardedSharpaControlClient(cfg.hard_joint_lower_rad, cfg.hard_joint_upper_rad,
        session_target_sdk_rad=q_goal, session_measured_target_sdk_rad=q_measured,
        container=cfg.sdk_container, sdk_root=cfg.sdk_root, sdk_python=cfg.sdk_python, serial=cfg.hand_serial,
        exact_policy_enabled=True, exact_policy_max_commands=args.cycles,
        exact_policy_lower_sdk_rad=lower, exact_policy_upper_sdk_rad=upper)
    phase, controller, index, previous_sequence = "preview", None, 0, 0
    q_command = q_goal.copy()
    log = (args.output / "control.jsonl").open("w")
    try:
        with deployment_resources(client) as resources:
            camera = resources.enter_context(RealSenseSession(RealSenseStreamConfig(serial=cfg.camera_serial, reset_on_start=False)))
            frame = camera.warmup(30)
            estimates = []
            for _ in range(100):
                frame = camera.next_frame()
                estimate = estimator.estimate(frame.color_bgr, camera.camera_matrix, camera.distortion_coefficients, timestamp_ms=frame.timestamp_ms)
                if estimate is not None:
                    estimates.append(estimate)
            calibration = aggregate_hand_camera_estimates(estimates, camera_matrix=camera.camera_matrix,
                distortion=camera.distortion_coefficients, estimator=estimator, capture_attempts=100)
            if not calibration.passed:
                raise RuntimeError(f"Camera calibration failed: {calibration.metrics}")
            (args.output / "calibration.json").write_text(json.dumps({"t_hand_camera": calibration.t_hand_camera.tolist(), "metrics": calibration.metrics}, indent=2))
            if mask.shape != frame.depth_m.shape or np.count_nonzero(mask) < 50:
                raise ValueError("Registration mask must match the camera resolution and contain the object")
            tracker.register(frame, camera_matrix=camera.camera_matrix, mask=mask > 0, iterations=8)
            watchdog = HandCameraDriftWatchdog(calibration.mean_t_camera_tag)
            capture = resources.enter_context(RealSenseCaptureWorker(camera))
            client.start()  # Reads state; the hand remains unarmed until the 'a' key.
            while True:
                tick = time.monotonic()
                if client.status.armed:
                    client.heartbeat()
                captured = capture.wait_for_frame(after_sequence=previous_sequence, timeout_s=0.1)
                if captured is None:
                    raise RuntimeError("Camera did not deliver a fresh frame")
                previous_sequence = captured.publish_sequence
                frame = captured.frame
                pose_camera_mesh, _ = tracker.track(frame, camera_matrix=camera.camera_matrix, iterations=2)
                pose = transform_to_pose(calibration.t_hand_camera @ pose_camera_mesh @ mesh_to_object)
                estimate = estimator.estimate(frame.color_bgr, camera.camera_matrix, camera.distortion_coefficients, timestamp_ms=frame.timestamp_ms)
                if watchdog.update(estimate).should_pause:
                    raise RuntimeError("Hand-camera calibration changed or AprilTag was lost")
                state = client.latest()
                now = time.monotonic()
                pose_age = now - captured.captured_monotonic_s
                if pose_age > 0.10:
                    raise RuntimeError("Tracked camera frame is stale")
                source_pose = initial["object_pose_palm"]
                position_error = (pose[:3] - source_pose[:3]).norm().item()
                angle_error = torch.rad2deg(2 * torch.acos(torch.clamp(torch.abs(torch.dot(pose[3:], source_pose[3:])), 0, 1))).item()
                pose_ready = position_error <= 0.01 and angle_error <= 15
                # Reject a missed control deadline before sending another motor target.
                if phase == "policy" and now - tick > 1 / 15:
                    raise RuntimeError("Control loop fell below 15 Hz before command")
                if phase in ("pregrasp", "close"):
                    if phase == "close" and not pose_ready:
                        client.hold()
                        phase = "held"
                    else:
                        endpoint = client.armed_pregrasp_target_sdk_rad if phase == "pregrasp" else q_goal
                        q_command = client.command_target(q_command + np.clip(endpoint - q_command, -0.01, 0.01), motion_phase=phase)
                elif phase == "policy":
                    command = controller.propose(state.angles_sdk_rad, pose, timestamp_ms=frame.timestamp_ms,
                        pose_age_s=pose_age, joint_age_s=now - state.received_monotonic_s)
                    client.command_exact_policy(command, exact_index=index, confirmation="EXACT_POLICY_CONTRACT")
                    controller.acknowledge(command)
                    q_command = command
                    index += 1
                    log.write(json.dumps({"cycle": index, "camera_timestamp_ms": frame.timestamp_ms, "q": state.angles_sdk_rad.tolist(), "q_cmd": command.tolist(), "pose": pose.tolist()}) + "\n")
                    log.flush()
                    if index == args.cycles:
                        client.hold()
                        phase = "complete"
                display = frame.color_bgr.copy()
                text = f"{phase} | grasp error {position_error*1000:.1f} mm / {angle_error:.1f} deg"
                cv2.putText(display, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(display, "a: arm/pregrasp  c: close  p: policy  SPACE: hold  q: quit", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow("POISE", display)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord("q")):
                    if client.status.armed:
                        client.normal_stop()
                    break
                if key == ord("a") and args.execute and phase == "preview":
                    q_command = client.arm(speed_coeff=cfg.speed_coeff, current_coeff=cfg.current_coeff, confirmation="POISE")
                    phase = "pregrasp"
                elif key == ord("c") and args.execute and phase == "pregrasp" and pose_ready:
                    phase = "close"
                elif key == ord("p") and args.execute and phase == "close" and pose_ready and np.max(np.abs(q_command - q_goal)) < 1e-6:
                    controller = PolicyController(bundle, state.angles_sdk_rad, q_command, pose, target_pose, gravity, timestamp_ms=frame.timestamp_ms)
                    phase = "policy"
                elif key == 32 and client.status.armed:
                    client.hold()
                    phase = "held"
                elapsed = time.monotonic() - tick
                if phase == "policy" and elapsed > 1 / 15:
                    raise RuntimeError("Control loop fell below 15 Hz")
                time.sleep(max(0, 0.05 - elapsed))
    finally:
        log.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
