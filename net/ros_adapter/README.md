# ROS2 Adapter

Converts the dynamics predictor's filtered EE target action into ROS2 `geometry_msgs/PoseStamped` messages so the project can drive Franka, KUKA, or any MoveIt-compatible manipulator.

The package has two layers:

- [`adapter.py`](adapter.py) -- pure-Python converter ([`EEActionToROS2`](adapter.py)). Handles unit scaling and quaternion convention; emits dataclass mirrors of the ROS2 wire format ([`messages.py`](messages.py)). No `rclpy` dependency, so it works in any inference / CI environment.
- [`node.py`](node.py) -- a real `rclpy` node ([`EEPoseBridge`](node.py)) that wraps the adapter and publishes on a configurable topic. Imported on demand; only loads in an environment with ROS2 installed.

## Conventions

| Quantity | Model output | ROS2 message |
| --- | --- | --- |
| Position units | sim units (`0.1 sim ~= 2 cm`) | meters |
| Quaternion order | `[qw, qx, qy, qz]` (scalar-first) | `[x, y, z, w]` (scalar-last) |
| Default frame | -- | `base_link` |

The default `sim_to_meter = 0.2` matches the scaling documented in the top-level [README](../../README.md) (`5.0 sim = 1 m`). Override on the `EEActionToROS2` constructor when working with already-metric inputs.

The 8th action dimension (gripper opening) is intentionally ignored on the pose channel; publish it on a separate gripper command topic that matches your hardware (e.g. `franka_gripper/move`).

## Pure-Python use

```python
from net.ros_adapter import EEActionToROS2

adapter = EEActionToROS2(frame_id="panda_link0")
target = pipeline.predict(batch)["ee_target_nxt"]   # torch.Tensor [B, 7] or [B, 8]
poses = adapter.to_pose_stamped(target)             # list[PoseStamped]
for p in poses:
    print(p.to_dict())
```

## With ROS2

```python
from net.ros_adapter.node import EEPoseBridge

bridge = EEPoseBridge(topic="/ee_target", frame_id="panda_link0")
for batch in loader:
    target = pipeline.predict(batch)["ee_target_nxt"]
    bridge.publish(target)
bridge.shutdown()
```

Smoke test from a sourced ROS2 shell:

```bash
python -m net.ros_adapter.node --topic /ee_target --frame panda_link0 --rate 10
# in another shell:
ros2 topic echo /ee_target
```

## Hardware integration notes

- **Franka (Cartesian impedance / `franka_cartesian_impedance_example_controller`)** -- consumes `geometry_msgs/PoseStamped` on `equilibrium_pose`. Set `frame_id="panda_link0"`.
- **MoveIt 2 `move_group`** -- accepts a `PoseStamped` goal via the `MoveGroup` action. Set `frame_id` to the planning frame (commonly `world` or the manipulator base).
- **MoveIt 2 Servo** -- subscribes to `PoseStamped` on the configured target-pose topic when `command_in_type: pose`. Pair with the project's safety filter so streamed targets are bounded.
- **KUKA (iiwa_ros2 / FRI)** -- the iiwa Cartesian controller exposes a `PoseStamped` interface analogous to Franka's; same adapter, change `frame_id`.

For mobile bases (Husky + manipulator), an additional Cartesian->base/manipulator split is needed; that is intentionally out of scope of this adapter.

## Tests

```bash
python -m pytest net/ros_adapter/tests
```

The tests cover unit scaling, quaternion order swap, batch handling, and defensive re-normalization. They have no torch / numpy dependency, so they run in any environment.
