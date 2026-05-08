## A Visual Predictive Model for Topological Manipulation of Deformable Objects


## Researcher — Navid Noorgostar

## Dependencies

### Create the conda environment
- `conda env create -f environment.yml`

### Install additional submodules
- `git submodule init && git submodule update`
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast): `cd net/nvdiffrast && pip install -e . && cd ../..`
- [sdftoolbox](https://github.com/cheind/sdftoolbox/): `cd sim/sdftoolbox && pip install -e . && cd ../..`

## Evaluation
Using the provided weights, the evaluation reproduces the main results from the paper. Note that due to dataset preprocessing and weights trained from scratch using this public code base, the results may vary slightly. Alternatively, train the model from scratch or create a new dataset as described below. Make sure to adapt the paths in the config files accordingly.
- `python net/prediction.py --config-name dyn "settings.test_only=True"`


## Training
Using the provided dataset, the autoencoder and the dynamics prediction are trained in two stages, as shown below. Alternatively, generate a custom dataset as described below.

Note that for multi-GPU training, e.g., using 2 GPUs, the `settings.ddp` flag needs to be set in the config. Run the scripts below with `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 {path_to_script}` instead of `python {path_to_script}`.

### Geometry-topology Autoencoder
- `python net/prediction.py --config-name ae`
- This saves the weights in the corresponding hydra output directory (i.e., `outputs/{date}/{time when run was started}/best.pth`). Either copy them to the default path (`weights/ae.pth`), or adapt the `settings.resume_path` in `net/config/dyn.yaml` accordingly before starting the next stage.

### Dynamics Prediction
- `python net/prediction.py --config-name dyn`
- Again, the weights are saved in the corresponding hydra output directory. Follow the directions above to make sure that `settings.test_path` points to the desired weights when running subsequent evaluations.



## Safety-Aware Action Filter

After the dynamics predictor proposes a next end-effector (EE) target, the predicted action is passed through a [SafetyAwareActionFilter](net/model/safety_filter.py) that projects it onto a safe set before it is consumed by the model. The filter checks five constraints in order:

1. **Joint / workspace limits** – clamps the EE xyz position to a configured workspace box and the gripper opening to its allowed range; the orientation quaternion is re-normalized.
2. **Velocity limits** – per-DoF caps on linear, angular and gripping velocities (mirrors the simulator's PID `vmax` settings in [sim/generate/config/ee/common.yaml](sim/generate/config/ee/common.yaml)).
3. **Base speed** – scalar cap on the magnitude of the translational velocity, treating the EE root as a mobile base.
4. **Action smoothness** – cap on linear acceleration between consecutive commanded targets to suppress jerky motion.
5. **Collision risk** – pulls the target back along its translation axis when the EE point cloud comes within a configured margin of the dough.

Thresholds are configured in [net/config/safety.yaml](net/config/safety.yaml). Set `safety.enabled=False` (or remove the `safety@safety` line from [net/config/common.yaml](net/config/common.yaml)) to bypass the filter.

The filter is applied transparently inside `Pipeline.predict` ([net/pipeline/pipeline.py:163](net/pipeline/pipeline.py#L163)): the predicted EE point cloud target is rigidly translated by the safety correction so the predictor only ever sees actions inside the safe set. Per-batch violation flags are stored under the `safety_info{postfix}` key of the returned data dictionary.

Tests for the filter live in [net/model/tests/test_safety_filter.py](net/model/tests/test_safety_filter.py) (per-check happy-path coverage) and [net/model/tests/test_safety_filter_extended.py](net/model/tests/test_safety_filter_extended.py) (boundary equality, mixed-batch independence, constraint ordering, idempotence, gradient flow, antipodal-quaternion handling, dtype propagation, and 7-D / 8-D shape preservation).

## Hardware Abstraction

The four supported deployment targets — `sim`, `franka_mobile`, `kuka_mobile`, and `husky_arm` — are described by a single [`RobotConfig`](net/hardware/robot_config.py) dataclass. Each kind ships as a built-in preset and a YAML file under [`net/config/robot/`](net/config/robot/), and consumers (the ROS2 adapter and the safety filter) read the same config so that changing the target robot only requires changing one selector:

```python
from net.hardware import load_robot_config
from net.ros_adapter import EEActionToROS2

cfg = load_robot_config("franka_mobile")            # or "kuka_mobile" / "husky_arm" / "sim"
adapter = EEActionToROS2.from_robot_config(cfg)     # uses cfg.base_frame_id and cfg.sim_to_meter
```

A `RobotConfig` captures: arm kind and DoF, base frame name, EE / gripper / base-velocity topics, sim-to-meter scaling, mobile-base presence, and a `safety` block of optional overrides on top of [`net/config/safety.yaml`](net/config/safety.yaml). Mobile-base kinds (`franka_mobile`, `kuka_mobile`, `husky_arm`) declare a `base_cmd_topic` for `geometry_msgs/Twist` commands and a wider workspace; the fixed-base `sim` kind keeps the tabletop defaults. YAML files may reuse a preset and override only the fields that differ:

```yaml
# my_franka_lab.yaml
kind: franka_mobile
ee_topic: /lab1/panda/equilibrium_pose
safety:
  v_lin_max: 0.3
```

Then `load_robot_config("path/to/my_franka_lab.yaml")` returns a config built on top of the `franka_mobile` preset. Tests live under [`net/hardware/tests/`](net/hardware/tests/) and cover preset registration, YAML round-trips, validation, safety-override merging, and the ROS2-adapter integration for all four kinds.

## ROS2 Integration

The filtered EE target action can be forwarded to a real robot via the [`net/ros_adapter`](net/ros_adapter/) package. It exposes:

- A pure-Python converter ([`EEActionToROS2`](net/ros_adapter/adapter.py)) that maps the predicted action (`[B, 7]` or `[B, 8]`, layout `[x, y, z, qw, qx, qy, qz, (open)]`) into `geometry_msgs/PoseStamped` dataclasses. It handles the sim-units → meters scaling (default `0.2`, matching the project convention) and the quaternion convention swap (model is scalar-first `[w, x, y, z]`; ROS2 is scalar-last `[x, y, z, w]`). No `rclpy` dependency, so it stays usable in any inference / CI environment.
- A first-class `rclpy` publisher ([`EEPoseBridge`](net/ros_adapter/node.py)) that wraps the converter and publishes on a configurable topic. Hard-tested against MoveIt 2 (`move_group` and Servo) and Franka Cartesian impedance interfaces; KUKA `iiwa_ros2` follows the same pattern with a different `frame_id`.
- A test suite under [`net/ros_adapter/tests/`](net/ros_adapter/tests/) covering unit scaling, quaternion order swap, batch handling, and defensive re-normalization.

Minimal usage from an inference loop:

```python
from net.ros_adapter import EEActionToROS2
adapter = EEActionToROS2(frame_id="panda_link0")
target = pipeline.predict(batch)["ee_target_nxt"]
poses = adapter.to_pose_stamped(target)   # list[PoseStamped]
```

See [`net/ros_adapter/README.md`](net/ros_adapter/README.md) for hardware-specific notes (Franka, KUKA, MoveIt Servo) and the full streaming example with `EEPoseBridge`.

### MoveIt2 / Twist / JointTrajectory export layer

For pipelines that need richer ROS2 message types than a single `PoseStamped`, [`MoveIt2Exporter`](net/ros_adapter/exporters.py) wraps the AC-DiT output into the four channels a real ROS2 stack actually consumes:

| Output | Use case | Method |
| --- | --- | --- |
| `geometry_msgs/PoseStamped` | direct Cartesian impedance / Servo target | `to_pose_stamped` |
| `MoveItPoseGoal` (subset of `moveit_msgs/MotionPlanRequest`) | MoveIt2 motion-plan request with planner knobs | `to_pose_goal` |
| `geometry_msgs/Twist` / `TwistStamped` | mobile-base `cmd_vel` | `to_twist`, `to_twist_stamped`, `twist_from_action_pair` |
| `trajectory_msgs/JointTrajectory` | joint-space waypoints (post-IK) | `to_joint_trajectory` |

Per-arm defaults — joint names, MoveIt group, end-effector link — come from the robot config, so swapping target hardware only changes the selector:

```python
from net.ros_adapter import MoveIt2Exporter

exporter = MoveIt2Exporter.from_robot_config("franka_mobile")

# 1) MoveIt2 motion-plan goal (Cartesian)
goal = exporter.to_pose_goal(action, max_velocity_scaling_factor=0.5)[0]
# goal.group_name == "panda_arm", goal.target_pose.header.frame_id == "panda_link0"

# 2) Twist for the mobile base, derived from two consecutive predictions
twist = exporter.twist_from_action_pair(prev_action, next_action, dt=0.1)

# 3) JointTrajectory after running IK (joint_positions: [T, J])
traj = exporter.to_joint_trajectory(joint_positions, dt=0.05)
# traj.joint_names == ["panda_joint1", ..., "panda_joint7"]
```

Tests live alongside the adapter in [`net/ros_adapter/tests/test_exporters.py`](net/ros_adapter/tests/test_exporters.py) and cover per-arm joint-name lookup, planner-knob validation, batch handling, finite-difference Twist derivation (translation, rotation, identity, negative dt), JointTrajectory width / time / count consistency, and end-to-end conversion of a single AC-DiT prediction into all three message types for every supported robot.

## Generation
Our simulation with topology annotation may be used to generate additional scenes or completely new datasets. 

To this end, first, derive novel scene definitions from `template.yaml`, e.g., by adapting `to_pos` and `to_quat` (grasp pose), or `close_d` (final opening width).

### Simulation
- `python sim/generate.py`
- This will create a `log.pkl` with particle-based information (and `visualization.gif` if `render=True` in config) in the scene directory.

### Processing
- `python sim/process.py`
- This will process the simulated scenes in parallel and create `data.h5` with additional mesh-based information.

