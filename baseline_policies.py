"""
Baseline Policies for Reward Probe Gates

Three baseline policies to validate reward functions:
1. Random Policy - uniform sampling from action space
2. Scripted Policy - hand-coded heuristic using IK
3. Oracle Policy - privileged state access with aggressive control

Usage:
    python baseline_policies.py --policy scripted --stage 3 --mode gui --episodes 5
"""

import argparse
import numpy as np
import pybullet as p
from collections import deque


class RandomPolicy:
    """
    Baseline: Random exploration.

    Expected Performance:
    - Stage 1 (Visibility): ~10-15% success
    - Stage 2-6: <5% success
    - Median episode return: -500 to -2000
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=False):
        """Sample random joint positions. SB3-compatible API."""
        action = self.action_space.sample()
        return action, None

    def reset(self):
        """No state to reset."""
        pass


class ScriptedPolicy:
    """
    Baseline: Hand-crafted PD controller using PyBullet IK.

    Stage-specific behaviors:
    - Stage 1: Random walk until cube visible
    - Stage 2-3: Move end-effector toward cube centroid (position only)
    - Stage 4-6: Add alignment correction (compute target orientation)

    Expected Performance:
    - Stage 1: ~45-55% success
    - Stage 2: ~30-40% success
    - Stage 3: ~25-35% success
    - Stage 4: ~15-25% success
    - Stage 5: ~10-20% success
    - Stage 6: ~5-10% success
    """

    def __init__(self, env, robot_id, end_effector_index, cube_id, stage=1):
        self.env = env
        self.robot_id = robot_id
        self.ee_index = end_effector_index
        self.cube_id = cube_id
        self.stage = stage

        # Proportional gain for velocity control
        self.kp = 1.5  # Moderate gain - too high causes overshoot

        # Random walk parameters for Stage 1
        self.random_walk_scale = 0.2
        self.cube_search_timeout = 50  # Steps before giving up
        self.search_counter = 0

        # Number of controllable joints
        self.num_joints = 7  # MyCobot has 7 joints total

    def predict(self, observation, deterministic=False):
        """
        Compute action based on stage and observation.

        observation = [ee_pos(3), cube_pos(3), joint_pos(7), distance(1), alignment(1), visible_flag(1)]
        """
        ee_pos = observation[0:3]
        cube_pos = observation[3:6]
        current_joint_pos = observation[6:13]  # 7 joints
        distance = observation[13]
        alignment = observation[14]
        visible_flag = observation[15]  # NEW: visibility flag (1.0 if visible, 0.0 if not)

        if self.stage == 1:
            # Stage 1: Random walk until cube visible
            # Check visibility flag from observation
            cube_visible = (visible_flag > 0.5)

            if cube_visible:
                # Cube found - hold still to maintain visibility
                action = np.zeros(self.num_joints)  # No movement
                self.search_counter = 0
            else:
                # Random exploration to find cube
                action = np.random.uniform(
                    -self.random_walk_scale,
                    self.random_walk_scale,
                    size=self.num_joints
                )
                self.search_counter += 1

        else:
            # Stage 2-6: Use IK to approach cube
            cube_visible = (visible_flag > 0.5)

            if not cube_visible:
                # Cube not visible: perform search behavior (random walk)
                action = np.random.uniform(
                    -self.random_walk_scale,
                    self.random_walk_scale,
                    size=self.num_joints
                )
                self.search_counter += 1
            else:
                # Cube visible: use IK-based tracking
                if self.stage >= 4:
                    # Stages 4-6: Include alignment correction
                    target_pos, target_orn = self._compute_aligned_target(
                        cube_pos, ee_pos, distance
                    )
                    target_joint_pos = p.calculateInverseKinematics(
                        self.robot_id,
                        self.ee_index,
                        target_pos,
                        targetOrientation=target_orn,
                        maxNumIterations=100,
                        residualThreshold=0.001
                    )
                else:
                    # Stages 2-3: Position only (no orientation)
                    target_pos = self._compute_approach_target(cube_pos, ee_pos, distance)
                    target_joint_pos = p.calculateInverseKinematics(
                        self.robot_id,
                        self.ee_index,
                        target_pos,
                        maxNumIterations=100,
                        residualThreshold=0.001
                    )

                # IK returns 6 values for joints 1-6 (revolute joints)
                # Joint 0 is FIXED (base_to_joint1), so we pad at START with 0
                target_joint_pos = np.array(target_joint_pos)
                if len(target_joint_pos) < self.num_joints:
                    # Pad at START: [0, ik1, ik2, ik3, ik4, ik5, ik6]
                    target_joint_pos = np.pad(target_joint_pos, (1, 0), constant_values=0)

                # Simple proportional control (no derivative - causes instability)
                action = self.kp * (target_joint_pos - current_joint_pos)
                self.search_counter = 0

        # Clip to action space bounds
        action = np.clip(action, -1.0, 1.0)

        # Debug disabled for production runs
        # if not hasattr(self, '_debug_printed'):
        #     print(f"[ScriptedPolicy Stage {self.stage}] First action: {action}")
        #     self._debug_printed = True

        return action, None

    def _check_cube_visible(self):
        """Get cube visibility from environment segmentation."""
        try:
            camera_data = self.env._get_camera_view()
            segmentation_image = camera_data[4]
            return np.any(segmentation_image == self.cube_id)
        except Exception as e:
            # Fallback if camera not available
            # For Stage 1: assume cube visible to use IK approach
            # For other stages: doesn't matter (always uses IK)
            return True  # Changed from False to avoid getting stuck

    def _compute_approach_target(self, cube_pos, ee_pos, distance):
        """
        Compute target position for approach (Stages 2-3).

        Strategy: Move 70% of the way toward cube (prevents overshoot).
        """
        direction = cube_pos - ee_pos
        approach_fraction = 0.7  # Conservative approach
        target_pos = ee_pos + approach_fraction * direction
        return target_pos

    def _compute_aligned_target(self, cube_pos, ee_pos, distance):
        """
        Compute target pose (position + orientation) for aligned approach (Stages 4-6).

        Strategy:
        1. Position: Same as _compute_approach_target
        2. Orientation: Align camera Z-axis to point at cube
        """
        # Position target
        target_pos = self._compute_approach_target(cube_pos, ee_pos, distance)

        # Orientation target: Camera Z-axis toward cube
        # Camera Z should point from ee to cube
        direction = cube_pos - ee_pos
        direction_norm = direction / (np.linalg.norm(direction) + 1e-6)

        # Construct rotation matrix with Z-axis = direction
        # X and Y axes perpendicular (choose arbitrary but consistent)
        z_axis = direction_norm

        # Choose arbitrary orthogonal X-axis (avoid singularity when z_axis is vertical)
        if abs(z_axis[2]) < 0.9:
            x_axis = np.array([z_axis[1], -z_axis[0], 0])
        else:
            x_axis = np.array([0, z_axis[2], -z_axis[1]])
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)

        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # Convert rotation matrix to quaternion using manual conversion
        # Since PyBullet doesn't have getQuaternionFromMatrix, compute manually
        target_orn = self._rotation_matrix_to_quaternion(rotation_matrix)

        return target_pos, target_orn

    def _rotation_matrix_to_quaternion(self, R):
        """
        Convert 3x3 rotation matrix to quaternion [x, y, z, w].
        Using Shepperd's method for numerical stability.
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return (x, y, z, w)

    def reset(self):
        """Reset policy state."""
        self.search_counter = 0


class OraclePolicy:
    """
    Oracle baseline: Privileged access to perfect simulator state.

    Key differences from Scripted:
    1. Always knows exact cube position (no perception failure)
    2. More aggressive gains (can trust state)
    3. Direct alignment from step 1 (no stage progression needed)
    4. Distance-adaptive approach to avoid collisions

    Expected Performance (stochastic mode):
    - Stage 1: ~85-95% success
    - Stage 2: ~70-80% success
    - Stage 3: ~60-75% success
    - Stage 4: ~55-70% success
    - Stage 5: ~50-65% success
    - Stage 6: ~40-55% success
    """

    def __init__(self, env, robot_id, end_effector_index, cube_id, stochastic=True):
        self.env = env
        self.robot_id = robot_id
        self.ee_index = end_effector_index
        self.cube_id = cube_id  # For privileged state access
        self.stochastic = stochastic  # Add small noise for realism

        # Proportional gain for velocity control (higher than scripted for faster response)
        self.kp = 2.0  # Slightly more aggressive than scripted

        # Noise parameters
        self.position_noise_std = 0.002  # 2mm position noise
        self.orientation_noise_std = 0.01  # Small orientation noise

        # Number of controllable joints
        self.num_joints = 7  # MyCobot has 7 joints total

    def predict(self, observation, deterministic=False):
        """
        Compute optimal action using PRIVILEGED state (ground truth cube position).

        Oracle policy always knows the true cube position via PyBullet query,
        regardless of visibility. This serves as an upper-bound baseline.

        observation = [ee_pos(3), cube_pos(3), joint_pos(7), distance(1), alignment(1), visible_flag(1)]
        """
        ee_pos = observation[0:3]
        current_joint_pos = observation[6:13]  # 7 joints

        # PRIVILEGED STATE: Get TRUE cube position from PyBullet (ignore masked observation)
        cube_pos_true, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos_true)

        # Compute true distance (for approach strategy)
        distance = np.linalg.norm(ee_pos - cube_pos)

        # Add stochastic noise if enabled (makes oracle more realistic)
        if self.stochastic and not deterministic:
            cube_pos = cube_pos + np.random.normal(0, self.position_noise_std, 3)

        # Compute target pose with alignment
        target_pos, target_orn = self._compute_optimal_target(
            cube_pos, ee_pos, distance
        )

        # Get target joint configuration via IK
        target_joint_pos = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_index,
            target_pos,
            targetOrientation=target_orn,
            maxNumIterations=200,  # More iterations for precision
            residualThreshold=0.0001
        )
        # IK returns 6 values for joints 1-6 (revolute joints)
        # Joint 0 is FIXED (base_to_joint1), so we pad at START with 0
        target_joint_pos = np.array(target_joint_pos)
        if len(target_joint_pos) < self.num_joints:
            # Pad at START: [0, ik1, ik2, ik3, ik4, ik5, ik6]
            target_joint_pos = np.pad(target_joint_pos, (1, 0), constant_values=0)

        # Simple proportional control (no derivative - causes instability)
        action = self.kp * (target_joint_pos - current_joint_pos)

        # Clip to action space
        action = np.clip(action, -1.0, 1.0)

        return action, None

    def _compute_optimal_target(self, cube_pos, ee_pos, distance):
        """
        Compute optimal target pose.

        Strategy: Exponentially approach cube based on current distance
        - Far away (>0.15m): Move 80% of the way
        - Medium (0.05-0.15m): Move 60% of the way
        - Close (<0.05m): Move 30% of the way (precision mode)
        """
        if distance > 0.15:
            approach_fraction = 0.8
        elif distance > 0.05:
            approach_fraction = 0.6
        else:
            approach_fraction = 0.3

        direction = cube_pos - ee_pos
        target_pos = ee_pos + approach_fraction * direction

        # Always compute aligned orientation
        direction_norm = direction / (np.linalg.norm(direction) + 1e-6)
        z_axis = direction_norm

        if abs(z_axis[2]) < 0.9:
            x_axis = np.array([z_axis[1], -z_axis[0], 0])
        else:
            x_axis = np.array([0, z_axis[2], -z_axis[1]])
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-6)

        y_axis = np.cross(z_axis, x_axis)

        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
        target_orn = self._rotation_matrix_to_quaternion(rotation_matrix)

        # Add orientation noise if stochastic
        if self.stochastic:
            orn_noise = np.random.normal(0, self.orientation_noise_std, 4)
            target_orn = np.array(target_orn) + orn_noise
            # Re-normalize quaternion
            target_orn = target_orn / (np.linalg.norm(target_orn) + 1e-6)
            target_orn = tuple(target_orn)

        return target_pos, target_orn

    def _rotation_matrix_to_quaternion(self, R):
        """
        Convert 3x3 rotation matrix to quaternion [x, y, z, w].
        Using Shepperd's method for numerical stability.
        """
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return (x, y, z, w)

    def reset(self):
        """Reset policy state (no state to reset for Oracle)."""
        pass


def run_policy_demo(policy_name='scripted', stage=3, mode='gui', episodes=5):
    """
    Demo script to visualize a baseline policy.

    Args:
        policy_name: 'random', 'scripted', or 'oracle'
        stage: 1-6
        mode: 'gui' or 'headless'
        episodes: number of episodes to run
    """
    # Import here to avoid circular dependency
    from curriculum_env import CurriculumCubeTrackingEnv

    # Create environment
    env = CurriculumCubeTrackingEnv(
        log_dir=f"baseline_demo_logs/stage{stage}",
        enable_curriculum=False,  # Fix stage
        mode=mode  # Will add this parameter to env
    )
    env.curriculum_manager.current_stage = stage

    # Create policy
    if policy_name == 'random':
        policy = RandomPolicy(env.action_space)
        print(f"\n=== Random Policy Demo - Stage {stage} ===")
    elif policy_name == 'scripted':
        policy = ScriptedPolicy(
            env.env,  # Unwrap to get C1 env
            env.env.robot_id,
            env.env.end_effector_index,
            env.env.cube_id,
            stage
        )
        print(f"\n=== Scripted Policy Demo - Stage {stage} ===")
    elif policy_name == 'oracle':
        policy = OraclePolicy(
            env.env,
            env.env.robot_id,
            env.env.end_effector_index,
            stochastic=True
        )
        print(f"\n=== Oracle Policy Demo - Stage {stage} ===")
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    # Run episodes
    episode_returns = []
    episode_successes = []

    for ep in range(episodes):
        obs = env.reset()
        policy.reset()
        done = False
        episode_return = 0
        step_count = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            episode_return += reward
            step_count += 1

        # Check success
        success = info.get('success', False)
        episode_returns.append(episode_return)
        episode_successes.append(success)

        print(f"  Episode {ep+1}/{episodes}: Return={episode_return:.1f}, "
              f"Distance={info['distance_to_cube']:.3f}m, "
              f"Alignment={info['alignment']:.2f}, "
              f"Visible={info['cube_visible']}, "
              f"Success={success}")

    # Summary
    print(f"\n  Mean Return: {np.mean(episode_returns):.1f}")
    print(f"  Success Rate: {np.mean(episode_successes):.1%}")
    print(f"  {'='*50}\n")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run baseline policy demo')
    parser.add_argument('--policy', type=str, default='scripted',
                       choices=['random', 'scripted', 'oracle'],
                       help='Policy to run')
    parser.add_argument('--stage', type=int, default=3,
                       choices=[1, 2, 3, 4, 5, 6],
                       help='Curriculum stage')
    parser.add_argument('--mode', type=str, default='gui',
                       choices=['gui', 'headless'],
                       help='GUI for visual verification, headless for speed')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')

    args = parser.parse_args()

    run_policy_demo(
        policy_name=args.policy,
        stage=args.stage,
        mode=args.mode,
        episodes=args.episodes
    )
