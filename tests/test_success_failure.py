import math

from g1_access_push.geometry.box_polygon import BoxGeometry
from g1_access_push.geometry.doorway import DoorGeometry
from g1_access_push.logging.episode_logger import EpisodeLogger
from g1_access_push.task.coordinate_frames import SE2Pose
from g1_access_push.task.success_failure import (
    FailureCode,
    FailureSignals,
    GoalPose,
    SuccessCriteria,
    TaskState,
    classify_failure,
    evaluate_success,
)


def _fixture() -> tuple[BoxGeometry, DoorGeometry, SuccessCriteria]:
    box = BoxGeometry(length=1.2, width=0.4, height=0.5)
    door = DoorGeometry(pose_world=SE2Pose(0.0, 0.0, 0.0), width=1.0, thickness=0.2)
    criteria = SuccessCriteria(
        goal=GoalPose(
            pose_world=SE2Pose(0.8, 0.0, 0.0),
            position_tolerance_m=0.02,
            yaw_tolerance_rad=math.radians(1.0),
        ),
        object_exit_margin_m=0.05,
        robot_rear_radius_m=0.30,
        robot_exit_margin_m=0.05,
    )
    return box, door, criteria


def test_success_requires_robot_and_object_to_clear_door() -> None:
    box, door, criteria = _fixture()
    object_only = TaskState(
        object_pose_world=SE2Pose(0.8, 0.0, 0.0),
        robot_base_pose_world=SE2Pose(0.2, 0.0, 0.0),
    )
    report = evaluate_success(object_only, box, door, criteria)
    assert report.object_clear_door
    assert not report.robot_clear_door
    assert not report.success

    both_clear = TaskState(
        object_pose_world=SE2Pose(0.8, 0.0, 0.0),
        robot_base_pose_world=SE2Pose(0.5, 0.0, 0.0),
    )
    report = evaluate_success(both_clear, box, door, criteria)
    assert report.success


def test_fall_or_collision_prevents_success() -> None:
    box, door, criteria = _fixture()
    fallen = TaskState(
        object_pose_world=SE2Pose(0.8, 0.0, 0.0),
        robot_base_pose_world=SE2Pose(0.5, 0.0, 0.0),
        fall=True,
    )
    assert not evaluate_success(fallen, box, door, criteria).success


def test_failure_codes_are_complete_and_primary_priority_is_fixed() -> None:
    report = classify_failure(
        FailureSignals(
            no_plan=True,
            contact_lost=True,
            fall=True,
            timeout=True,
        )
    )
    assert report.primary == FailureCode.F5_FALL_OR_TORQUE
    assert report.all_codes == (
        FailureCode.F1_NO_PLAN,
        FailureCode.F4_CONTACT_OR_RESPONSE,
        FailureCode.F5_FALL_OR_TORQUE,
        FailureCode.F8_TIMEOUT,
    )


def test_episode_logger_persists_failure_code(tmp_path) -> None:
    logger = EpisodeLogger(
        root_dir=tmp_path,
        episode_id="stage0_case_001",
        metadata={"seed": 7, "wbc_id": "not_connected"},
    )
    logger.write_resolved_config({"door": {"width_m": 1.0}})
    logger.append_step(0.0, {"event": "reset"})
    episode_dir = logger.finalize(
        {"success": False, "failure_code": FailureCode.F1_NO_PLAN}
    )

    summary = EpisodeLogger.load_json(episode_dir / "summary.json")
    steps = EpisodeLogger.load_steps(episode_dir)
    assert summary["failure_code"] == FailureCode.F1_NO_PLAN.value
    assert steps == [{"event": "reset", "timestamp_s": 0.0}]
