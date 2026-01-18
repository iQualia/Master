"""
Stage-specific reward functions for curriculum learning.
Based on user's C1.py implementation with progressive complexity.

User's original C1.py reward (Stage 2):
    reward = -distance²
    if cube_visible: reward += 10
    else: reward -= 10

This module builds on that design, adding stages before and after.
"""

import numpy as np
from typing import Tuple, Dict


def compute_stage1_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """
    Stage 1: Visibility ONLY - Learn camera control.

    This is SIMPLER than user's C1.py - just visibility.
    Goal: Agent learns to keep cube in view before worrying about distance.

    Args:
        distance: Euclidean distance from end-effector to cube (unused in this stage)
        alignment: Camera-cube alignment score (unused in this stage)
        cube_visible: Whether cube is in camera view
        collision_count: Number of collision points detected (unused)

    Returns:
        reward: Total reward value
        info: Dictionary with reward components
    """
    reward = 0.0
    info = {}

    # ONLY reward visibility (user's ±10 values)
    if cube_visible:
        visibility_reward = 10.0
        reward += visibility_reward
        info['visibility_reward'] = visibility_reward
    else:
        visibility_penalty = -10.0
        reward += visibility_penalty
        info['visibility_reward'] = visibility_penalty

    info['distance_reward'] = 0.0  # Not used in this stage
    info['alignment_reward'] = 0.0  # Not used in this stage
    info['total_reward'] = reward

    return reward, info


def compute_stage2_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """
    Stage 2: Visibility + Easy Distance with Temporal Tolerance.

    Key innovation: Allows temporary visibility loss if distance improving.
    Implements "blind moment for better positioning" strategy.

    This bridges the gap between Stage 1 (visibility only) and Stage 3
    (tighter distance requirements) by:
    1. Introducing distance control with easier 0.22m threshold
    2. Reducing penalty for temporary blindness during approach
    3. Rewarding visibility recovery after risky movements

    Args:
        distance: Current distance to cube (meters)
        alignment: Current alignment score (unused in this stage)
        cube_visible: Whether cube currently visible
        collision_count: Collision count (unused)
        prev_distance: Previous step's distance (for approach detection)
        visibility_history: Deque of last N visibility states

    Returns:
        reward: Total reward value
        info: Dictionary with reward components
    """
    reward = 0.0
    info = {}

    # Distance penalty (same as old Stage 2)
    distance_reward = -np.square(distance)
    reward += distance_reward
    info['distance_reward'] = distance_reward

    # TEMPORAL VISIBILITY REWARD - Key innovation
    if cube_visible:
        visibility_reward = 10.0
        reward += visibility_reward
        info['visibility_reward'] = visibility_reward
        info['blind_approach'] = False

        # Bonus for regaining visibility after risky approach
        if visibility_history is not None and len(visibility_history) >= 3:
            # Check if was blind in last 5 steps
            was_recently_blind = sum(1 for v in list(visibility_history)[-5:] if not v) > 0
            if was_recently_blind and distance < 0.22:  # Recovered AND close
                recovery_bonus = 3.0
                reward += recovery_bonus
                info['recovery_bonus'] = recovery_bonus
            else:
                info['recovery_bonus'] = 0.0
        else:
            info['recovery_bonus'] = 0.0
    else:
        # Check if approaching despite blindness
        is_approaching = (prev_distance is not None and
                        distance < prev_distance - 0.015)  # 1.5cm improvement threshold

        if is_approaching:
            # REDUCED penalty for blind approach (-3 instead of -10)
            visibility_penalty = -3.0
            info['blind_approach'] = True
            info['approach_rate'] = prev_distance - distance if prev_distance else 0.0
        else:
            # Full penalty if not making progress
            visibility_penalty = -10.0
            info['blind_approach'] = False
            info['approach_rate'] = 0.0

        reward += visibility_penalty
        info['visibility_reward'] = visibility_penalty
        info['recovery_bonus'] = 0.0

    info['alignment_reward'] = 0.0
    info['collision_penalty'] = 0.0
    info['total_reward'] = reward

    return reward, info


def compute_stage3_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """
    Stage 3: Visibility + Medium Distance (0.18m) - Was old Stage 2.

    This is EXACTLY what user has in C1.py lines 345-356.
    Goal: Approach cube while maintaining visibility.

    Args:
        distance: Euclidean distance from end-effector to cube
        alignment: Camera-cube alignment score (unused in this stage)
        cube_visible: Whether cube is in camera view
        collision_count: Number of collision points detected (unused)

    Returns:
        reward: Total reward value
        info: Dictionary with reward components
    """
    reward = 0.0
    info = {}

    # Distance penalty (USER'S EXACT FORMULA from C1.py:345)
    distance_reward = -np.square(distance)  # -distance²
    reward += distance_reward
    info['distance_reward'] = distance_reward

    # Visibility reward/penalty (USER'S EXACT FORMULA from C1.py:354-356)
    if cube_visible:
        visibility_reward = 10.0
        reward += visibility_reward
        info['visibility_reward'] = visibility_reward
    else:
        visibility_penalty = -10.0
        reward += visibility_penalty
        info['visibility_reward'] = visibility_penalty

    info['alignment_reward'] = 0.0  # User commented this out
    info['collision_penalty'] = 0.0  # User commented this out
    info['total_reward'] = reward

    return reward, info


def compute_stage4_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """
    Stage 4: Add PARTIAL Alignment - Gradually introduce orientation (was old Stage 3).

    Enables user's commented alignment reward (C1.py:346) but at HALF strength.
    Goal: Start learning alignment without overwhelming the agent.

    Args:
        distance: Euclidean distance from end-effector to cube
        alignment: Camera-cube alignment score 0-1
        cube_visible: Whether cube is in camera view
        collision_count: Number of collision points detected (unused)

    Returns:
        reward: Total reward value
        info: Dictionary with reward components
    """
    reward = 0.0
    info = {}

    # Keep user's distance formula
    distance_reward = -np.square(distance)
    reward += distance_reward
    info['distance_reward'] = distance_reward

    # Add PARTIAL alignment (half of user's intended weight)
    # User's commented code: reward += alignment * 10
    # We use: alignment * 5 (easier to learn)
    if cube_visible:
        alignment_reward = alignment * 5.0  # Half strength
        reward += alignment_reward
        info['alignment_reward'] = alignment_reward

        visibility_reward = 10.0
        reward += visibility_reward
        info['visibility_reward'] = visibility_reward
    else:
        visibility_penalty = -10.0
        reward += visibility_penalty
        info['visibility_reward'] = visibility_penalty
        info['alignment_reward'] = 0.0  # No alignment if can't see cube

    info['collision_penalty'] = 0.0
    info['total_reward'] = reward

    return reward, info


def compute_stage5_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """
    Stage 5: FULL Alignment - Uncomment user's original design (was old Stage 4).

    This enables what user WANTED but couldn't train (C1.py:346).
    Goal: Master complete tracking task with full alignment.

    Args:
        distance: Euclidean distance from end-effector to cube
        alignment: Camera-cube alignment score 0-1
        cube_visible: Whether cube is in camera view
        collision_count: Number of collision points detected (unused)

    Returns:
        reward: Total reward value
        info: Dictionary with reward components
    """
    reward = 0.0
    info = {}

    # User's distance formula
    distance_reward = -np.square(distance)
    reward += distance_reward
    info['distance_reward'] = distance_reward

    # FULL alignment (user's intended weight from C1.py:346)
    # reward += alignment * 10  # Now enabled!
    if cube_visible:
        alignment_reward = alignment * 10.0  # Full strength
        reward += alignment_reward
        info['alignment_reward'] = alignment_reward

        visibility_reward = 10.0
        reward += visibility_reward
        info['visibility_reward'] = visibility_reward
    else:
        visibility_penalty = -10.0
        reward += visibility_penalty
        info['visibility_reward'] = visibility_penalty
        info['alignment_reward'] = 0.0

    info['collision_penalty'] = 0.0  # Still not using collisions
    info['total_reward'] = reward

    return reward, info


def compute_stage6_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """
    Stage 6: SPARSE Rewards - Publishable final policy (was old Stage 5).

    Reduces dense shaping, adds large sparse bonus for success.
    Goal: Create robust policy that doesn't rely on dense rewards.

    Args:
        distance: Euclidean distance from end-effector to cube
        alignment: Camera-cube alignment score 0-1
        cube_visible: Whether cube is in camera view
        collision_count: Number of collision points detected

    Returns:
        reward: Total reward value
        info: Dictionary with reward components
    """
    reward = 0.0
    info = {}

    # Minimal distance shaping (80% reduction)
    distance_reward = -np.square(distance) * 0.2  # Much weaker
    reward += distance_reward
    info['distance_reward'] = distance_reward

    # Large SPARSE bonus for achieving success criteria
    # Success: distance < 0.05 AND alignment > 0.8 AND visible
    if cube_visible and distance < 0.05 and alignment > 0.8:
        success_bonus = 100.0  # Big reward for complete success
        reward += success_bonus
        info['success_bonus'] = success_bonus
    else:
        # Small penalty for not achieving full criteria
        failure_penalty = -5.0
        reward += failure_penalty
        info['success_bonus'] = failure_penalty

    # Still track visibility
    if cube_visible:
        info['visibility_reward'] = 5.0  # Smaller visibility bonus
        reward += 5.0
    else:
        info['visibility_reward'] = -5.0
        reward -= 5.0

    # Alignment only matters if visible
    if cube_visible:
        alignment_reward = alignment * 10.0
        reward += alignment_reward
        info['alignment_reward'] = alignment_reward
    else:
        info['alignment_reward'] = 0.0

    # Optionally enable collision penalty (user commented this out)
    if collision_count > 0:
        collision_penalty = -10.0 * collision_count
        reward += collision_penalty
        info['collision_penalty'] = collision_penalty
    else:
        info['collision_penalty'] = 0.0

    info['total_reward'] = reward

    return reward, info


def get_reward_function(stage: int):
    """
    Get reward function for a given curriculum stage.

    Args:
        stage: Curriculum stage (1-6)

    Returns:
        Reward function for that stage

    Raises:
        ValueError: If stage is not 1-6
    """
    functions = {
        1: compute_stage1_reward,
        2: compute_stage2_reward,  # NEW: Temporal tolerance
        3: compute_stage3_reward,
        4: compute_stage4_reward,
        5: compute_stage5_reward,
        6: compute_stage6_reward
    }

    if stage not in functions:
        raise ValueError(f"Invalid stage {stage}, must be 1-6")

    return functions[stage]


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("REWARD FUNCTIONS TEST - Based on User's C1.py Design")
    print("="*70)

    # Test scenario
    test_distance = 0.10  # 10cm from cube
    test_alignment = 0.7  # 70% aligned
    test_visible = True
    test_collisions = 0

    print(f"\nTest scenario:")
    print(f"  Distance: {test_distance:.2f}m")
    print(f"  Alignment: {test_alignment:.2f}")
    print(f"  Cube visible: {test_visible}")
    print(f"  Collisions: {test_collisions}")

    print("\n" + "="*70)
    print("Reward values across curriculum stages:")
    print("="*70)

    for stage in range(1, 6):
        reward_fn = get_reward_function(stage)
        reward, info = reward_fn(test_distance, test_alignment, test_visible, test_collisions)

        print(f"\nStage {stage}:")
        print(f"  Total reward: {reward:+.2f}")
        for key, value in info.items():
            if key != 'total_reward':
                print(f"    {key}: {value:+.2f}")

    # Test Stage 2 matches user's C1.py exactly
    print("\n" + "="*70)
    print("Verification: Stage 2 should match user's C1.py")
    print("="*70)

    reward_stage2, info_stage2 = compute_stage2_reward(
        test_distance, test_alignment, test_visible, test_collisions
    )

    expected_reward = -np.square(test_distance) + 10.0  # User's formula
    print(f"\nStage 2 reward: {reward_stage2:.4f}")
    print(f"Expected (from C1.py): {expected_reward:.4f}")
    print(f"Match: {'✓ YES' if abs(reward_stage2 - expected_reward) < 0.001 else '✗ NO'}")

    # Test sparse reward (Stage 5)
    print("\n" + "="*70)
    print("Sparse reward test (Stage 5)")
    print("="*70)

    # Close enough for success
    close_distance = 0.04
    good_alignment = 0.85

    reward_success, info_success = compute_stage5_reward(
        close_distance, good_alignment, True, 0
    )

    # Too far for success
    far_distance = 0.15
    reward_fail, info_fail = compute_stage5_reward(
        far_distance, good_alignment, True, 0
    )

    print(f"\nSuccess case (d=0.04, a=0.85, visible): {reward_success:+.2f}")
    print(f"  Contains +100 bonus: {'✓' if info_success.get('success_bonus', 0) > 0 else '✗'}")

    print(f"\nFailure case (d=0.15, a=0.85, visible): {reward_fail:+.2f}")
    print(f"  No +100 bonus: {'✓' if info_fail.get('success_bonus', 0) < 0 else '✗'}")

    print("\n" + "="*70)
    print("✓ All tests complete!")
    print("="*70)
