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

## Generation
Our simulation with topology annotation may be used to generate additional scenes or completely new datasets. 

To this end, first, derive novel scene definitions from `template.yaml`, e.g., by adapting `to_pos` and `to_quat` (grasp pose), or `close_d` (final opening width).

### Simulation
- `python sim/generate.py`
- This will create a `log.pkl` with particle-based information (and `visualization.gif` if `render=True` in config) in the scene directory.

### Processing
- `python sim/process.py`
- This will process the simulated scenes in parallel and create `data.h5` with additional mesh-based information.

