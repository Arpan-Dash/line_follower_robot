# line_follower_robot

ROS 2 / Gazebo camera line follower for the RISC Induction Hackathon track worlds.

## Run a supplied trial world

Build and source the workspace, then pass any SDF from the competition repository:

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch line_follower_bot simulation.py \
  world:=/path/to/RISC-INDUCTION-HACKATHON-26/TRIAL_PATHS/hexagon_0_5.sdf
```

The controller publishes:

- `/cmd_vel`: velocity command sent to Gazebo
- `/line_follower/error`: normalized camera error in `[-1, 1]` (negative is left)
- `/line_follower/confidence`: detection confidence
- `/line_follower/state`: `tracking`, `recovering`, or `waiting_for_camera`
- `/line_follower/debug_image`: camera image annotated with ROI, detected center, error, and angular command

The detection uses a multi-row ROI, adaptive/fixed threshold agreement, morphology,
confidence-weighted centroiding, speed scheduling, filtered PID control, and directional
line-loss recovery. It uses no external navigation, map, or machine-learning dependency.
