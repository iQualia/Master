# Training Session Recap - January 22, 2026

## Summary
Curriculum learning for PyBullet visual servoing with SAC (MyCobot 280 robot arm).
All root causes identified and fixed. Ready for fresh training.

---

## FIXES IMPLEMENTED TODAY

### 1. Learning Rate Override (CRITICAL)
**File:** `train_curriculum.py`

When loading SAC from checkpoint, now properly overrides optimizer learning rates:
```python
model.learning_rate = learning_rate
model.lr_schedule = lambda _: learning_rate
for param_group in model.actor.optimizer.param_groups:
    param_group['lr'] = learning_rate
```

### 2. Replay Buffer Clearing on Stage Advance (CRITICAL)
**Files:** `curriculum_manager.py`, `curriculum_env.py`

Buffer is now cleared when curriculum advances to prevent stale reward data:
```python
def advance_stage(self, model=None):
    if model and hasattr(model, 'replay_buffer'):
        model.replay_buffer.reset()
```

### 3. Monotonically Composable Rewards (HIGH)
**File:** `reward_functions.py`

Completely redesigned reward function:
- Each stage ADDS components, never removes
- Stage 1: Visibility only (+10/-10)
- Stage 2+: Add distance shaping (+0.5 - d²)
- Stage 4+: Add alignment bonus (gradually increasing weight)
- Stage 6: Add sparse success bonus (+20 ON TOP of existing)

**Verified monotonic:**
```
Stage 1: +10.00
Stage 2: +10.49 (adds distance)
Stage 3: +10.49
Stage 4: +11.99 (adds alignment)
Stage 5: +13.49
Stage 6: +34.99 (adds success bonus)
```

### 4. Removed Stage 2 Temporal Tolerance (MEDIUM)
**File:** `reward_functions.py`

The "blind approach" tolerance (-3 instead of -10) was unique to Stage 2 and caused confusion. Removed for consistency.

### 5. Added Periodic Review Mechanism (MEDIUM)
**File:** `curriculum_env.py`

Every 500 episodes, agent practices previous stage for 50 episodes:
- Prevents catastrophic forgetting
- Only activates when at Stage 2 or higher
- Uses appropriate reward function for review stage

---

## ROOT CAUSES FIXED

| # | Issue | Fix | Status |
|---|-------|-----|--------|
| 1 | Reward discontinuity | Monotonically composable rewards | ✅ FIXED |
| 2 | Stale replay buffer | Clear buffer on stage advance | ✅ FIXED |
| 3 | LR not applied on resume | Override optimizer LRs | ✅ FIXED |
| 4 | Stage 2 temporal tolerance | Removed | ✅ FIXED |
| 5 | Stage 6 sparse cliff | Bonus on top, not replacement | ✅ FIXED |
| 6 | No periodic review | Added review mechanism | ✅ FIXED |

---

## TRAINING COMMAND

**Start fresh training (do NOT resume old checkpoints):**
```bash
source /home/Master/.venv/bin/activate
python train_curriculum.py \
  --total_timesteps 2000000 \
  --run_name training_run_5 \
  --learning_rate 0.0001 \
  --checkpoint_freq 25000
```

**Why fresh?** Old checkpoints have:
- Stale reward data in replay buffer
- Rewards from old (non-monotonic) reward function

---

## FILES MODIFIED

| File | Changes |
|------|---------|
| `train_curriculum.py` | LR override, model ref to env |
| `curriculum_manager.py` | Buffer reset in advance_stage() |
| `curriculum_env.py` | set_model(), periodic review |
| `reward_functions.py` | Complete rewrite (monotonic) |

---

## CURRICULUM STAGES (Updated)

| Stage | Reward Components | Success Criteria |
|-------|-------------------|------------------|
| 1 | Visibility (±10) | 70% visibility |
| 2 | + Distance shaping | visibility + d < 0.35m |
| 3 | + Distance shaping | visibility + d < 0.25m |
| 4 | + Alignment (×2.5) | visibility + d < 0.20m + a > 0.3 |
| 5 | + Alignment (×5.0) | visibility + d < 0.15m + a > 0.3 |
| 6 | + Alignment (×7.5) + Success bonus | visibility + d < 0.12m + a > 0.5 |

---

## PREVIOUS SESSION FIXES (Already Committed)

1. Fixed Broken Replay Buffer Loading
2. Added `--restore_curriculum` CLI Argument
3. Fixed Checkpoint Directory Bug
4. Fixed Uninitialized `stats` Variable
5. Added JSON Validation
6. Added NaN Sanitization
7. Fixed Division by Zero
8. Added Action NaN Check

---

## Git Status

**Branch:** `robust-training-setup`

```bash
cd /home/Master
source .venv/bin/activate
git status
```
