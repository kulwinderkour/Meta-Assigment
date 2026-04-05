from __future__ import annotations

from environment.graders.grader_easy import (
    _doctor_specialty,
    _no_conflicts,
    _slot_start,
    _slot_window,
)
from environment.models import RewardBreakdown, SchedulerState


class HardGrader:
    """
    Grader for the hard task (8 patients, emergency escalation, rescheduling trap).

    Scoring breakdown
    -----------------
    coverage        (0.30) : fraction of 8 patients scheduled
    specialty_match (0.25) : fraction booked with the correct doctor specialty
                              (p007 going to cardiology is penalised)
    urgency_order   (0.25) : composite score across 3 sub-checks:
                                • p004 flagged as emergency       → 0.50 of this component
                                • p001 before p002 in cardiology  → 0.25 of this component
                                • p004 before p003 in neurology   → 0.25 of this component
    no_conflicts    (0.10) : no two patients share a slot
    time_constraints(0.10) : fraction of scheduled patients in preferred window

    Correct specialty map
    ---------------------
    p001 → cardiology     p002 → cardiology
    p003 → neurology      p004 → neurology
    p005 → orthopedics    p006 → dermatology
    p007 → general practice   p008 → general practice
    """

    _W = {
        "coverage":         0.30,
        "specialty_match":  0.25,
        "urgency_order":    0.25,
        "no_conflicts":     0.10,
        "time_constraints": 0.10,
    }

    _CORRECT_SPECIALTIES = {
        "p001": "cardiology",
        "p002": "cardiology",
        "p003": "neurology",
        "p004": "neurology",
        "p005": "orthopedics",
        "p006": "dermatology",
        "p007": "general practice",
        "p008": "general practice",
    }

    _PREFERRED_WINDOWS = {
        "p001": "morning",
        "p002": "morning",
        "p003": "morning",
        "p004": "morning",
        "p005": "morning",
        "p006": "afternoon",
        "p007": "afternoon",
        "p008": "morning",
    }

    # (higher-urgency patient, lower-urgency patient) within same specialty
    _PRIORITY_PAIRS = [
        ("p001", "p002"),  # cardiology: critical > high
        ("p004", "p003"),  # neurology:  critical (after flag) > medium
    ]

    def grade(self, state: SchedulerState) -> RewardBreakdown:
        scheduled = state.scheduled
        n = len(self._CORRECT_SPECIALTIES)  # 8

        # --- coverage ---
        coverage = len(scheduled) / n

        # --- specialty_match ---
        hits = sum(
            1
            for pid, expected in self._CORRECT_SPECIALTIES.items()
            if pid in scheduled
            and _doctor_specialty(state, scheduled[pid]).lower() == expected.lower()
        )
        specialty_match = hits / n

        # --- urgency_order (composite) ---
        urgency_order = self._check_urgency_order(state, scheduled)

        # --- no_conflicts ---
        no_conflicts = _no_conflicts(scheduled)

        # --- time_constraints ---
        tc_denom = sum(1 for pid in self._PREFERRED_WINDOWS if pid in scheduled)
        tc_hits  = sum(
            1
            for pid, pref in self._PREFERRED_WINDOWS.items()
            if pid in scheduled and _slot_window(state, scheduled[pid]) == pref
        )
        time_constraints = (tc_hits / tc_denom) if tc_denom > 0 else 1.0

        w = self._W
        total = (
              coverage         * w["coverage"]
            + specialty_match  * w["specialty_match"]
            + urgency_order    * w["urgency_order"]
            + no_conflicts     * w["no_conflicts"]
            + time_constraints * w["time_constraints"]
        )

        return RewardBreakdown(
            total            = round(min(1.0, total), 4),
            coverage         = round(coverage, 4),
            urgency_order    = round(urgency_order, 4),
            specialty_match  = round(specialty_match, 4),
            no_conflicts     = round(no_conflicts, 4),
            time_constraints = round(time_constraints, 4),
        )

    def _check_urgency_order(
        self, state: SchedulerState, scheduled: dict
    ) -> float:
        """
        Composite urgency/emergency score in [0.0, 1.0]:

          Sub-check 1 (weight 0.50):
            p004 must be present in state.emergency_flags.

          Sub-check 2 (weight 0.25):
            p001 (critical) must hold an earlier cardiology slot than p002 (high).
            Partial credit (0.125) if p001 is in cardiology but p002 is absent.

          Sub-check 3 (weight 0.25):
            p004 (critical after flag) must hold an earlier neurology slot than p003 (medium).
            Partial credit (0.125) if p004 is in neurology but p003 is absent.
        """
        score = 0.0

        # Sub-check 1: emergency flag
        if "p004" in state.emergency_flags:
            score += 0.50

        # Sub-checks 2 & 3: same-specialty priority pairs
        pair_weights = {
            ("p001", "p002"): ("cardiology", 0.25),
            ("p004", "p003"): ("neurology",  0.25),
        }
        for (higher, lower), (expected_spec, weight) in pair_weights.items():
            if higher not in scheduled:
                continue  # higher-priority patient not scheduled — no credit

            higher_spec = _doctor_specialty(state, scheduled[higher])
            if higher_spec.lower() != expected_spec.lower():
                continue  # higher-priority patient in wrong specialty

            if lower not in scheduled:
                score += weight / 2  # partial credit — no competition to lose to
                continue

            lower_spec = _doctor_specialty(state, scheduled[lower])
            if lower_spec.lower() != expected_spec.lower():
                score += weight / 2  # lower patient in wrong specialty — higher wins by default
                continue

            h_start = _slot_start(state, scheduled[higher])
            l_start = _slot_start(state, scheduled[lower])
            if h_start <= l_start:
                score += weight  # full credit — correct ordering

        return round(min(1.0, score), 4)
