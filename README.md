# Safety-Aware Manipulation of Deformable Objects

**A visual predictive model for topological manipulation of deformable objects — extended into a deployable robotics stack with safety filtering, uncertainty-aware planning, closed-loop replanning and ROS 2 integration.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ROS 2](https://img.shields.io/badge/ROS%202-22314E?logo=ros&logoColor=white)](https://docs.ros.org/)
[![MoveIt 2](https://img.shields.io/badge/MoveIt%202-0A7CFF)](https://moveit.ros.org/)
[![Tests](https://img.shields.io/badge/tests-passing-4CAF50)](#testing)
[![License](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)](LICENSE)

---

Predicting how dough, cloth or tissue will deform under a gripper is only half the problem. The other
half is what a robot is allowed to **do** with that prediction: whether the commanded action is safe,
how much the model actually disagrees with itself, when to stop and look again, and how the result
reaches a real arm.

This repository closes that loop. A geometry–topology autoencoder and dynamics predictor propose the
next end-effector target; every proposal is then projected onto a safe set, scored for uncertainty,
folded into a receding-horizon controller, and exported as ROS 2 messages for Franka, KUKA or a mobile
manipulator.


## Highlights

| Module | What it does | Where |
|---|---|---|
| **Safety-aware action filter** | Projects each predicted action onto a safe set through five ordered constraints, before the predictor ever sees it | [`net/model/safety_filter.py`](net/model/safety_filter.py) |
| **Uncertainty-aware prediction** | Ensemble sampling with four disagreement scores; gates the planning horizon when the model is unsure | [`net/uncertainty/`](net/uncertainty/) |
| **Closed-loop replanning** | Receding-horizon control that re-observes each step — **15× lower final error** than open-loop under drift | [`net/control/`](net/control/) |
| **Hardware abstraction** | One `RobotConfig` selector covers `sim`, `franka_mobile`, `kuka_mobile`, `husky_arm` | [`net/hardware/`](net/hardware/) |
| **ROS 2 / MoveIt 2 export** | `PoseStamped`, MoveIt motion-plan goals, `Twist` for mobile bases, `JointTrajectory` after IK | [`net/ros_adapter/`](net/ros_adapter/) |
| **Ablation framework** | Declarative studies producing paper-ready Markdown/CSV tables; every constraint shown to contribute | [`net/ablation/`](net/ablation/) |

## Pipeline

```
   observation
        │
        ▼
 Geometry–Topology Autoencoder  ──►  latent
        │
        ▼
   Dynamics Prediction  ──►  proposed EE action  [x y z qw qx qy qz (open)]
        │                              │
        │                              ▼
        │                 SafetyAwareActionFilter
        │                 joint · velocity · base speed
        │                 smoothness · collision margin
        │                              │
        ▼                              ▼
 Uncertainty Ensemble  ─────►  Closed-Loop Replanner
 N samples → score              predict H, execute K, re-observe
        │                              │
        └── score > threshold ─────────┘
            truncate horizon,
            look again sooner
                                       │
                                       ▼
                        ROS 2 / MoveIt 2 export layer
                     Franka · KUKA · Husky · simulation
```

## Contents

[Quick start](#quick-start) · [Training](#training) · [Evaluation](#evaluation) ·
[Safety filter](#safety-aware-action-filter) · [Uncertainty](#uncertainty-aware-action-prediction) ·
[Closed-loop replanning](#closed-loop-replanning) · [Hardware](#hardware-abstraction) ·
[ROS 2](#ros-2-integration) · [Ablations](#ablation-studies) · [Dataset](#dataset-generation) ·
[Testing](#testing) · [Citation](#citation)

---

## Quick start

```bash
# 1. environment
conda env create -f environment.yml
conda activate <env-name>

# 2. submodules
git submodule init && git submodule update
cd net/nvdiffrast && pip install -e . && cd ../..    # NVlabs/nvdiffrast
cd sim/sdftoolbox && pip install -e . && cd ../..    # cheind/sdftoolbox

# 3. run with the provided weights
python net/prediction.py --config-name dyn "settings.test_only=True"
```

Two things you can run immediately, without trained weights:

```bash
python -m net.ablation.run --study safety_filter     # safety-constraint ablation table
python -m net.ablation.run --study robot_workspace   # workspace tightness sweep
```

## Training

Two stages. Adapt the paths in the config files before switching stages.

```bash
python net/prediction.py --config-name ae     # 1. geometry–topology autoencoder
python net/prediction.py --config-name dyn    # 2. dynamics prediction
```

Weights land in the hydra output directory (`outputs/{date}/{time}/best.pth`). Either copy them to the
default path (`weights/ae.pth`) or point `settings.resume_path` in `net/config/dyn.yaml` at them before
starting stage 2 — likewise `settings.test_path` for later evaluation.

Multi-GPU (two GPUs shown): set `settings.ddp` in the config and launch with

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 net/prediction.py --config-name dyn
```

## Evaluation

```bash
python net/prediction.py --config-name dyn "settings.test_only=True"
```

Reproduces the main results using the provided weights (place them in `weights/{ae,dyn}.pth`). Because of dataset preprocessing and weights
trained from scratch on this public code base, numbers may vary slightly from the published ones.

---

## Safety-aware action filter

After the dynamics predictor proposes a next end-effector target, the action passes through
[`SafetyAwareActionFilter`](net/model/safety_filter.py), which projects it onto a safe set **before the
model consumes it**. Five constraints are applied in order:

| # | Constraint | Effect |
|---|---|---|
| 1 | **Joint / workspace limits** | Clamps EE xyz to a configured workspace box and gripper opening to its range; re-normalises the orientation quaternion |
| 2 | **Velocity limits** | Per-DoF caps on linear, angular and gripping velocity, mirroring the simulator's PID `vmax` settings |
| 3 | **Base speed** | Scalar cap on translational velocity magnitude, treating the EE root as a mobile base |
| 4 | **Action smoothness** | Caps linear acceleration between consecutive targets, suppressing jerky motion |
| 5 | **Collision risk** | Pulls the target back along its translation axis when the EE point cloud enters a configured margin around the object |

Thresholds live in [`net/config/safety.yaml`](net/config/safety.yaml). Bypass with
`safety.enabled=False`, or remove the `safety@safety` line from
[`net/config/common.yaml`](net/config/common.yaml).

The filter is applied transparently inside `Pipeline.predict`
([`net/pipeline/pipeline.py:163`](net/pipeline/pipeline.py#L163)): the predicted EE point-cloud target is
rigidly translated by the safety correction, so the predictor only ever observes actions inside the safe
set. Per-batch violation flags are returned under the `safety_info{postfix}` key.

## Uncertainty-aware action prediction

[`net/uncertainty/`](net/uncertainty/) adds ensemble sampling and a small library of uncertainty scores
over predicted action trajectories. The interface is model-agnostic — anything yielding one candidate
trajectory per call works, whether that is MC dropout, a model ensemble or a diffusion sampler.

```python
from net.uncertainty import ensemble_predict, mean_std_score

def sample_fn(observation, horizon, goal, sample_id):
    # sample_id seeds MC dropout, selects the ensemble member,
    # or seeds the diffusion sampler — caller's choice
    return predict_one_trajectory(observation, horizon, goal, seed=sample_id)

result = ensemble_predict(sample_fn, observation, horizon=8, n_samples=16)
# result.mean_actions [T, D] · std_actions [T, D] · samples [N, T, D] · score (scalar)
```

Four scores ship, depending on what "uncertain" should mean:

| Score | Use |
|---|---|
| `mean_std_score` | Average disagreement — default, robust to outliers |
| `max_std_score` | Worst-case disagreement — gate-friendly |
| `disagreement_score` | Mean pairwise sample distance — no Gaussian assumption |
| `diffusion_entropy_score` | Differential entropy of a per-step Gaussian fit, mirroring how diffusion models report uncertainty through the noise schedule |

### Uncertainty-gated replanning

`make_uncertainty_aware_predict_fn` drops the ensemble straight into
[`run_closed_loop`](net/control/replanner.py). The controller receives the ensemble mean as its action
sequence, and an optional gate truncates the horizon whenever the score exceeds a threshold — forcing
the robot to re-observe sooner precisely when its own model disagrees with itself.

```python
from net.uncertainty import make_uncertainty_aware_predict_fn
from net.control import ReplannerConfig, run_closed_loop

scores = []
predict_fn = make_uncertainty_aware_predict_fn(
    sample_fn,
    n_samples=8,
    score_threshold=0.05,   # truncate above 5 cm equivalent disagreement
    short_horizon=2,
    record_to=scores,       # per-call score log
)
res = run_closed_loop(predict_fn, world, initial_obs,
                      config=ReplannerConfig(horizon=8, execute_steps=1), goal=goal)
```

## Closed-loop replanning

[`net/control/`](net/control/) wraps the dynamics predictor into a receding-horizon controller: rather
than predicting one long sequence and executing it blind, it re-observes and asks for a fresh horizon.

```python
from net.control import ReplannerConfig, run_closed_loop, run_open_loop, compare_loops

cfg = ReplannerConfig(horizon=8, execute_steps=1, max_steps=100, goal_tolerance=0.01)

result   = run_closed_loop(predict_fn, execute_fn, initial_obs, config=cfg, goal=goal)
baseline = run_open_loop(predict_fn, execute_fn, initial_obs, horizon=100, goal=goal)
cmp      = compare_loops(predict_fn, lambda: build_world(seed=0), initial_obs, config=cfg, goal=goal)
```

The interface is plain callables — `predict_fn(obs, horizon, goal) -> Sequence[action]` and
`execute_fn(obs, action) -> (new_obs, info)` — so the same orchestrator drives synthetic fixtures and a
real model on hardware. `execute_steps=1` is canonical MPC; setting it to `horizon` collapses to
open-loop.

### Why closed-loop wins

Measured on identical noise schedules ([`net/control/tests/`](net/control/tests/)):

| Scenario | Open-loop error | Closed-loop error | Improvement |
|---|---:|---:|---:|
| Constant drift (+0.3/step, 30 steps) | 9.00 | 0.60 | **15×** |
| Gaussian noise (σ = 0.3, 30 steps, 20 seeds) | 1.24 | 0.28 | **4.5×** |

The drift case has a clean P-control story: the closed-loop controller absorbs the bias each step, while
the open-loop predictor's internal model never gets the chance to react. The Gaussian case is the
statistical version of the same argument — closed-loop variance is bounded by a geometric series in the
controller gain, whereas open-loop variance grows linearly with the horizon.

## Hardware abstraction

Four deployment targets — `sim`, `franka_mobile`, `kuka_mobile`, `husky_arm` — are described by one
[`RobotConfig`](net/hardware/robot_config.py) dataclass. Both the ROS 2 adapter and the safety filter
read the same config, so retargeting hardware means changing one selector:

```python
from net.hardware import load_robot_config
from net.ros_adapter import EEActionToROS2

cfg     = load_robot_config("franka_mobile")     # or kuka_mobile / husky_arm / sim
adapter = EEActionToROS2.from_robot_config(cfg)  # uses cfg.base_frame_id, cfg.sim_to_meter
```

A `RobotConfig` captures arm kind and DoF, base frame, EE/gripper/base-velocity topics, sim-to-metre
scaling, mobile-base presence, and a `safety` block overriding
[`net/config/safety.yaml`](net/config/safety.yaml). Mobile kinds declare a `base_cmd_topic` for
`geometry_msgs/Twist` and a wider workspace; fixed-base `sim` keeps tabletop defaults. YAML files may
extend a preset and override only what differs:

```yaml
# my_franka_lab.yaml
kind: franka_mobile
ee_topic: /lab1/panda/equilibrium_pose
safety:
  v_lin_max: 0.3
```

## ROS 2 integration

[`net/ros_adapter`](net/ros_adapter/) forwards the filtered action to a real robot:

- **[`EEActionToROS2`](net/ros_adapter/adapter.py)** — pure-Python converter mapping the predicted action
  (`[B, 7]` or `[B, 8]`, layout `[x, y, z, qw, qx, qy, qz, (open)]`) into `geometry_msgs/PoseStamped`
  dataclasses. Handles sim-units → metres scaling and the quaternion convention swap (model is
  scalar-first, ROS 2 is scalar-last). No `rclpy` dependency, so it runs in any inference or CI
  environment.
- **[`EEPoseBridge`](net/ros_adapter/node.py)** — a real `rclpy` publisher wrapping the converter. Tested
  against MoveIt 2 (`move_group` and Servo) and Franka Cartesian impedance interfaces; KUKA `iiwa_ros2`
  follows the same pattern with a different `frame_id`.

```python
from net.ros_adapter import EEActionToROS2
adapter = EEActionToROS2(frame_id="panda_link0")
poses   = adapter.to_pose_stamped(pipeline.predict(batch)["ee_target_nxt"])
```

### MoveIt 2 / Twist / JointTrajectory export

[`MoveIt2Exporter`](net/ros_adapter/exporters.py) wraps the model output into the four channels a real
ROS 2 stack consumes:

| Output | Use case | Method |
|---|---|---|
| `geometry_msgs/PoseStamped` | Cartesian impedance / Servo target | `to_pose_stamped` |
| `MoveItPoseGoal` | MoveIt 2 motion-plan request with planner knobs | `to_pose_goal` |
| `geometry_msgs/Twist` | Mobile-base `cmd_vel` | `to_twist`, `twist_from_action_pair` |
| `trajectory_msgs/JointTrajectory` | Joint-space waypoints, post-IK | `to_joint_trajectory` |

```python
from net.ros_adapter import MoveIt2Exporter
exporter = MoveIt2Exporter.from_robot_config("franka_mobile")

goal  = exporter.to_pose_goal(action, max_velocity_scaling_factor=0.5)[0]
twist = exporter.twist_from_action_pair(prev_action, next_action, dt=0.1)
traj  = exporter.to_joint_trajectory(joint_positions, dt=0.05)
```

Per-arm defaults — joint names, MoveIt group, end-effector link — come from the robot config, so
swapping hardware changes only the selector. See
[`net/ros_adapter/README.md`](net/ros_adapter/README.md) for hardware-specific notes.

## Ablation studies

[`net/ablation/`](net/ablation/) is a declarative framework, so that "every constraint contributes" is a
reproducible claim rather than an assertion. A study is a `(base_config, list[AblationConfig], seed)`
triple; `study.run(evaluator)` returns a table rendering as paper-ready Markdown or CSV.

```bash
python -m net.ablation.run --study safety_filter
python -m net.ablation.run --study robot_workspace --format csv --out results.csv
python -m net.ablation.run --study model_ablation      # NaN cells until a runner is registered
```

Example output (seed 0), abbreviated:

| Ablation | joint | velocity | base speed | smoothness | collision | mean correction |
|---|---:|---:|---:|---:|---:|---:|
| `full` | 0.844 | 1.000 | 1.000 | 0.594 | 0.500 | 0.313 |
| `no_joint_limits` | **0.000** | 1.000 | 0.996 | 0.504 | 0.500 | 0.304 |
| `no_velocity_limits` | 0.789 | **0.000** | 1.000 | 0.746 | 0.500 | 0.309 |
| `no_base_speed` | 0.777 | 1.000 | **0.000** | 1.000 | 0.500 | 0.304 |
| `no_smoothness` | 0.777 | 1.000 | 1.000 | **0.000** | 0.500 | 0.277 |
| `no_collision` | 0.805 | 1.000 | 1.000 | 0.582 | **0.000** | 0.303 |
| `disabled` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Each `no_X` row drops only its own constraint to zero while the others keep firing on the same seeded
data. **The zero diagonal is the evidence** that each check contributes independently.

Three studies ship: `safety_filter_study` (five constraints plus a disabled lower bound),
`robot_workspace_study` (workspace tightness sweep), and `model_ablation_study` (3D input,
mobility-to-body conditioning, safety filter — scaffolded, returning `NaN` until you register a runner
with `set_model_runner`, so the table layout is reproducible without weights on the current machine).

Custom studies are a one-liner:

```python
from net.ablation import AblationConfig, AblationStudy, safety_filter_evaluator

study = AblationStudy(
    name="my_collision_sweep",
    base_config={...},
    ablations=[AblationConfig(f"margin_{m}mm", overrides={"collision_margin": m / 1000})
               for m in (1, 5, 10)],
    seed=0,
)
print(study.run(safety_filter_evaluator).to_markdown())
```

## Dataset generation

The simulation with topology annotation can generate additional scenes or entirely new datasets. Derive
scene definitions from `template.yaml`, adapting `to_pos` and `to_quat` (grasp pose) or `close_d` (final
opening width).

```bash
python sim/generate.py   # → log.pkl with particle information (+ visualization.gif if render=True)
python sim/process.py    # → data.h5 with mesh-based information, processed in parallel
```

## Testing

Every module ships with tests, and they encode behaviour rather than just coverage:

| Suite | What it locks in |
|---|---|
| [`net/model/tests/`](net/model/tests/) | Per-constraint happy paths; boundary equality, mixed-batch independence, constraint ordering, idempotence, gradient flow, antipodal quaternions, dtype propagation, 7-D/8-D shape preservation |
| [`net/uncertainty/tests/`](net/uncertainty/tests/) | 40 tests: byte-level determinism, zero score for deterministic samplers, empirical std scaling linearly with injected noise (within ~20 % of the 2× ratio), `max_std ≥ mean_std` always, gate firing exactly at threshold, loud failure on malformed `sample_fn` output |
| [`net/control/tests/`](net/control/tests/) | The drift and Gaussian-noise comparisons above, on fixed seeds |
| [`net/hardware/tests/`](net/hardware/tests/) | Preset registration, YAML round-trips, validation, safety-override merging, adapter integration for all four robot kinds |
| [`net/ros_adapter/tests/`](net/ros_adapter/tests/) | Unit scaling, quaternion order swap, batch handling, finite-difference Twist derivation, JointTrajectory consistency, end-to-end conversion for every supported robot |
| [`net/ablation/tests/`](net/ablation/tests/) | Declaration-order stability, per-row seed independence, deterministic reruns, strict-mode key validation, Markdown/CSV column alignment |

```bash
pytest net/
```

## Citation

```bibtex
@software{noorgostar_safety_aware_manipulation,
  author = {Noorgostar, Navid},
  title  = {Safety-Aware Manipulation of Deformable Objects},
  year   = {2026},
  url    = {https://github.com/Navid-Noorgostar-Research/safety-aware-manipulation-ros2}
}
```

Please also cite the underlying predictive model:

```bibtex
@inproceedings{bauer2024doughnet,
  title     = {DoughNet: A Visual Predictive Model for Topological Manipulation of Deformable Objects},
  author    = {Bauer, Dominik and Xu, Zhenjia and Song, Shuran},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {92--108},
  year      = {2024},
  doi       = {10.1007/978-3-031-72940-9_6}
}
```

## Acknowledgements

This work builds directly on **[DoughNet](https://dough-net.github.io/)** by Dominik Bauer, Zhenjia Xu
and Shuran Song (Columbia University and Stanford University), presented at ECCV 2024 —
[paper](https://link.springer.com/chapter/10.1007/978-3-031-72940-9_6) ·
[project page](https://dough-net.github.io/) · [code](https://github.com/dornik/doughnet).
The denoising autoencoder, the autoregressive latent dynamics model and the topology-annotated
simulator are theirs; this repository adds the deployment layer around them.

Also built on [nvdiffrast](https://github.com/NVlabs/nvdiffrast) (NVlabs) and
[sdftoolbox](https://github.com/cheind/sdftoolbox) (cheind).

## Contact

**Navid Noorgostar** — navid.noorgostar22@gmail.com

Contact modelling and grasp planning for deformable objects · learning-based manipulation with
simulation-to-reality transfer · safety filtering of commanded actions.

## License

See [LICENSE](LICENSE).
