"""Small helpers shared by adaptive curriculum implementations."""

from __future__ import annotations


def adaptive_soft_promotion_is_eligible(
    *,
    current_step: int,
    last_level_change_step: int,
    last_global_soft_promotion_step: int,
    plateau_steps: int,
    global_cooldown_steps: int,
    success_rate: float,
    success_threshold: float,
    history_batches: int,
    min_history_batches: int,
    frontier_done_total: float,
    min_frontier_done: int,
) -> bool:
    """Return whether one curriculum axis may use its plateau escape step."""

    return (
        current_step - last_level_change_step >= plateau_steps
        and current_step - last_global_soft_promotion_step >= global_cooldown_steps
        and success_rate >= success_threshold
        and history_batches >= min_history_batches
        and frontier_done_total >= min_frontier_done
    )
