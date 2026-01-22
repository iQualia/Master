"""
Stage-specific reward functions for curriculum learning.
Based on user's C1.py implementation with progressive complexity.

REDESIGNED: Rewards are now MONOTONICALLY COMPOSABLE.
Each stage ADDS reward components, never removes or changes existing ones.
This prevents catastrophic forgetting when curriculum advances.

User's original C1.py reward (Stage 2):
    reward = -distance²
    if cube_visible: reward += 10
    else: reward -= 10
"""

import numpy as np
from typing import Tuple, Dict


def _sanitize_inputs(distance: float, alignment: float) -> Tuple[float, float]:
    """Sanitize inputs to prevent NaN/Inf propagation in rewards."""
    # Handle NaN/Inf distance
    if not np.isfinite(distance):
        distance = 0.5  # Default to mid-range distance
    distance = np.clip(distance, 0.0, 1.0)

    # Handle NaN/Inf alignment
    if not np.isfinite(alignment):
        alignment = 0.0
    alignment = np.clip(alignment, -1.0, 1.0)

    return distance, alignment


def compute_unified_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    stage: int,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """
    UNIFIED reward function with stage-based weights.

    CRITICAL DESIGN PRINCIPLE: Each stage ADDS components, never removes.
    This ensures Q-values remain compatible across stage transitions.

    Stage progression:
    - Stage 1: Visibility only (±10)
    - Stage 2+: Add distance penalty (-distance²)
    - Stage 4+: Add alignment bonus (gradually increasing weight)
    - Stage 6: Add sparse success bonus (ON TOP of existing rewards)

    Args:
        distance: Euclidean distance from end-effector to cube
        alignment: Camera-cube alignment score (0-1)
        cube_visible: Whether cube is in camera view
        stage: Current curriculum stage (1-6)
        collision_count: Number of collisions (unused)
        prev_distance: Previous distance (unused in unified version)
        visibility_history: Recent visibility states (unused in unified version)

    Returns:
        reward: Total reward value
        info: Dictionary with reward components
    """
    distance, alignment = _sanitize_inputs(distance, alignment)

    reward = 0.0
    info = {}

    # =========================================================================
    # COMPONENT 1: VISIBILITY (ALL STAGES - same weight always)
    # =========================================================================
    if cube_visible:
        visibility_reward = 10.0
    else:
        visibility_reward = -10.0
    reward += visibility_reward
    info['visibility_reward'] = visibility_reward

    # =========================================================================
    # COMPONENT 2: DISTANCE SHAPING (Stage 2+)
    # Rewards getting closer (positive when close, zero when far)
    # This ensures monotonicity: being close is BETTER in later stages
    # =========================================================================
    if stage >= 2:
        # Max distance penalty is -1.0 (at d=1.0), bonus is 0 at d=0
        # We shift by +0.5 to make it positive when close
        # At d=0: reward = +0.5, at d=0.5: reward = +0.25, at d=1.0: reward = -0.5
        distance_shaping = 0.5 - np.square(distance)
        reward += distance_shaping
        info['distance_reward'] = distance_shaping
    else:
        info['distance_reward'] = 0.0

    # =========================================================================
    # COMPONENT 3: ALIGNMENT BONUS (Stage 4+, gradually increasing weight)
    # Only counts when cube is visible (can't align to what you can't see)
    # =========================================================================
    if stage >= 4 and cube_visible:
        # Gradually increase alignment weight across stages
        align_weight = {4: 2.5, 5: 5.0, 6: 7.5}.get(stage, 7.5)
        alignment_reward = alignment * align_weight
        reward += alignment_reward
        info['alignment_reward'] = alignment_reward
    else:
        info['alignment_reward'] = 0.0

    # =========================================================================
    # COMPONENT 4: SPARSE SUCCESS BONUS (Stage 6 only)
    # ADDS to existing rewards, doesn't replace them
    # This is a bonus ON TOP of the dense shaping, not instead of it
    # =========================================================================
    if stage == 6:
        # Success criteria: close + aligned + visible
        if cube_visible and distance < 0.12 and alignment > 0.5:
            success_bonus = 20.0  # Bonus on top, not replacement
            reward += success_bonus
            info['success_bonus'] = success_bonus
        else:
            info['success_bonus'] = 0.0
    else:
        info['success_bonus'] = 0.0

    # No collision penalty (user disabled this)
    info['collision_penalty'] = 0.0
    info['total_reward'] = reward

    return reward, info


# =============================================================================
# WRAPPER FUNCTIONS FOR EACH STAGE (for backward compatibility)
# All call the unified function with the appropriate stage number
# =============================================================================

def compute_stage1_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """Stage 1: Visibility ONLY."""
    return compute_unified_reward(distance, alignment, cube_visible, stage=1,
                                   collision_count=collision_count,
                                   prev_distance=prev_distance,
                                   visibility_history=visibility_history)


def compute_stage2_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """Stage 2: Visibility + Distance."""
    return compute_unified_reward(distance, alignment, cube_visible, stage=2,
                                   collision_count=collision_count,
                                   prev_distance=prev_distance,
                                   visibility_history=visibility_history)


def compute_stage3_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """Stage 3: Visibility + Distance (same as Stage 2, tighter success criteria)."""
    return compute_unified_reward(distance, alignment, cube_visible, stage=3,
                                   collision_count=collision_count,
                                   prev_distance=prev_distance,
                                   visibility_history=visibility_history)


def compute_stage4_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """Stage 4: Add partial alignment (weight 2.5)."""
    return compute_unified_reward(distance, alignment, cube_visible, stage=4,
                                   collision_count=collision_count,
                                   prev_distance=prev_distance,
                                   visibility_history=visibility_history)


def compute_stage5_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """Stage 5: Stronger alignment (weight 5.0)."""
    return compute_unified_reward(distance, alignment, cube_visible, stage=5,
                                   collision_count=collision_count,
                                   prev_distance=prev_distance,
                                   visibility_history=visibility_history)


def compute_stage6_reward(
    distance: float,
    alignment: float,
    cube_visible: bool,
    collision_count: int = 0,
    prev_distance: float = None,
    visibility_history: 'deque' = None
) -> Tuple[float, Dict]:
    """Stage 6: Full alignment (weight 7.5) + sparse success bonus."""
    return compute_unified_reward(distance, alignment, cube_visible, stage=6,
                                   collision_count=collision_count,
                                   prev_distance=prev_distance,
                                   visibility_history=visibility_history)


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
        2: compute_stage2_reward,
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
    print("REWARD FUNCTIONS TEST - MONOTONICALLY COMPOSABLE DESIGN")
    print("="*70)

    # Test scenario
    test_distance = 0.10  # 10cm from cube
    test_alignment = 0.6  # 60% aligned
    test_visible = True
    test_collisions = 0

    print(f"\nTest scenario:")
    print(f"  Distance: {test_distance:.2f}m")
    print(f"  Alignment: {test_alignment:.2f}")
    print(f"  Cube visible: {test_visible}")

    print("\n" + "="*70)
    print("Reward values across curriculum stages:")
    print("="*70)
    print("\nCRITICAL: Rewards should be MONOTONICALLY INCREASING across stages")
    print("(same state gives higher reward as stages progress)\n")

    prev_reward = float('-inf')
    for stage in range(1, 7):
        reward, info = compute_unified_reward(
            test_distance, test_alignment, test_visible, stage, test_collisions
        )

        # Check monotonicity
        monotonic = "✓" if reward >= prev_reward else "✗ PROBLEM!"
        prev_reward = reward

        print(f"Stage {stage}: {reward:+7.2f} {monotonic}")
        for key, value in info.items():
            if key != 'total_reward' and value != 0:
                print(f"    {key}: {value:+.2f}")

    # Test invisible case
    print("\n" + "="*70)
    print("Test: Cube NOT visible (should give negative rewards)")
    print("="*70)

    for stage in range(1, 7):
        reward, info = compute_unified_reward(
            test_distance, test_alignment, False, stage, test_collisions
        )
        print(f"Stage {stage}: {reward:+7.2f}")

    # Test Stage 6 success bonus
    print("\n" + "="*70)
    print("Test: Stage 6 success bonus")
    print("="*70)

    # Near success (just below threshold)
    near_d, near_a = 0.11, 0.49
    near_reward, _ = compute_unified_reward(near_d, near_a, True, 6)
    print(f"Near success (d={near_d}, a={near_a}): {near_reward:+.2f}")

    # Success (above threshold)
    success_d, success_a = 0.10, 0.55
    success_reward, _ = compute_unified_reward(success_d, success_a, True, 6)
    print(f"Success (d={success_d}, a={success_a}): {success_reward:+.2f}")

    bonus_diff = success_reward - near_reward
    print(f"Bonus for crossing threshold: {bonus_diff:+.2f}")
    print(f"(Should be +20 from success bonus + small alignment diff)")

    print("\n" + "="*70)
    print("✓ All tests complete!")
    print("="*70)
