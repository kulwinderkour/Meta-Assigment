from __future__ import annotations

from environment.models import RewardBreakdown, SchedulerState


class EasyGrader:
    """
    Grader for the easy task (1 patient, neurology appointment).

    Scoring breakdown
    -----------------
    coverage        (0.35) : was p001 scheduled at all?
    specialty_match (0.40) : was the booked doctor a neurologist?
    time_constraints(0.15) : was the slot in the patient's preferred morning window?
    urgency_order   (0.05) : N/A for 1 patient — always 1.0
    no_conflicts    (0.05) : N/A for 1 patient — always 1.0
    """

    _W = {
        "coverage":         0.35,
        "specialty_match":  0.40,
        "time_constraints": 0.15,
        "urgency_order":    0.05,
        "no_conflicts":     0.05,
    }

    def grade(self, state: SchedulerState) -> RewardBreakdown:
        scheduled = state.scheduled

        # --- coverage ---
        coverage = 1.0 if "p001" in scheduled else 0.0

        # --- specialty_match ---
        specialty_match = 0.0
        if "p001" in scheduled:
            spec = _doctor_specialty(state, scheduled["p001"])
            if spec.lower() == "neurology":
                specialty_match = 1.0

        # --- time_constraints ---
        # Full credit for morning (preferred); half credit for correct specialty
        # but wrong time window; zero for wrong specialty.
        time_constraints = 0.0
        if "p001" in scheduled:
            window = _slot_window(state, scheduled["p001"])
            if specialty_match == 1.0:
                time_constraints = 1.0 if window == "morning" else 0.5
            # wrong specialty → no time-constraint credit

        # --- trivially perfect for single-patient episode ---
        urgency_order = 1.0
        no_conflicts  = 1.0

        w = self._W
        total = (
            coverage         * w["coverage"]
            + specialty_match  * w["specialty_match"]
            + time_constraints * w["time_constraints"]
            + urgency_order    * w["urgency_order"]
            + no_conflicts     * w["no_conflicts"]
        )

        return RewardBreakdown(
            total            = round(min(1.0, total), 4),
            coverage         = round(coverage, 4),
            urgency_order    = round(urgency_order, 4),
            specialty_match  = round(specialty_match, 4),
            no_conflicts     = round(no_conflicts, 4),
            time_constraints = round(time_constraints, 4),
        )


# ---------------------------------------------------------------------------
# Shared helpers (module-level so grader_medium / grader_hard can import them)
# ---------------------------------------------------------------------------

def _doctor_specialty(state: SchedulerState, slot_id: str) -> str:
    """Return the specialty of the doctor holding `slot_id`."""
    for sl in state.all_slots:
        if sl.slot_id == slot_id:
            for doc in state.all_doctors:
                if doc.doctor_id == sl.doctor_id:
                    return doc.specialty
    return ""


def _slot_window(state: SchedulerState, slot_id: str) -> str:
    """Return the time_window of `slot_id`."""
    for sl in state.all_slots:
        if sl.slot_id == slot_id:
            return sl.time_window
    return ""


def _slot_start(state: SchedulerState, slot_id: str) -> tuple:
    """Return (date, start_time) for `slot_id` — lexicographically comparable."""
    for sl in state.all_slots:
        if sl.slot_id == slot_id:
            return (sl.date, sl.start_time)
    return ("9999", "99:99")


def _no_conflicts(scheduled: dict) -> float:
    """1.0 if no two patients share a slot, else 0.0."""
    vals = list(scheduled.values())
    return 1.0 if len(vals) == len(set(vals)) else 0.0
