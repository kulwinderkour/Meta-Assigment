from __future__ import annotations

from environment.graders.grader_easy import (
    _doctor_specialty,
    _no_conflicts,
    _slot_start,
    _slot_window,
)
from environment.models import RewardBreakdown, SchedulerState


class MediumGrader:
    """
    Grader for the medium task (4 patients, scarce cardiology slots).

    Scoring breakdown
    -----------------
    coverage        (0.35) : fraction of 4 patients scheduled
    specialty_match (0.35) : fraction booked with the correct doctor specialty
    urgency_order   (0.20) : p001 (critical/cardiology) holds the earlier slot
                              than any other cardiology patient — binary 0/1
    no_conflicts    (0.05) : no two patients share a slot
    time_constraints(0.05) : fraction of scheduled patients in their preferred window

    Correct specialty map
    ---------------------
    p001 → cardiology       p002 → orthopedics
    p003 → dermatology      p004 → general practice
    """

    _W = {
        "coverage":         0.35,
        "specialty_match":  0.35,
        "urgency_order":    0.20,
        "no_conflicts":     0.05,
        "time_constraints": 0.05,
    }

    _CORRECT_SPECIALTIES = {
        "p001": "cardiology",
        "p002": "orthopedics",
        "p003": "dermatology",
        "p004": "general practice",
    }

    _PREFERRED_WINDOWS = {
        "p001": "morning",
        "p002": "morning",
        "p003": "afternoon",
        "p004": "afternoon",
    }

    def grade(self, state: SchedulerState) -> RewardBreakdown:
        scheduled = state.scheduled
        n = len(self._CORRECT_SPECIALTIES)  # 4

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

        # --- urgency_order ---
        # p001 (critical) must hold the EARLIER cardiology slot than p002 (high).
        # Score 1.0 if p001's (date, start_time) ≤ p002's; 0.0 otherwise.
        # If either patient is not scheduled we cannot confirm ordering → 0.0.
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
        Verify that p001 (critical) has the earlier cardiology slot vs p002 (high).

        Returns 1.0 only when:
          - both p001 and p002 are scheduled
          - both land in cardiology slots
          - p001's (date, start_time) ≤ p002's (date, start_time)

        Partial credit (0.5) when p001 is correctly in cardiology but p002 is
        absent (no competition to lose priority to).
        """
        if "p001" not in scheduled:
            return 0.0

        p001_spec = _doctor_specialty(state, scheduled["p001"])
        if p001_spec.lower() != "cardiology":
            return 0.0  # p001 in wrong specialty — urgency ordering irrelevant

        if "p002" not in scheduled:
            return 0.5  # p001 in cardiology but no p002 to compare against

        p002_spec = _doctor_specialty(state, scheduled["p002"])
        if p002_spec.lower() != "cardiology":
            return 0.5  # p002 not in cardiology — p001 automatically has priority

        p001_start = _slot_start(state, scheduled["p001"])
        p002_start = _slot_start(state, scheduled["p002"])
        return 1.0 if p001_start <= p002_start else 0.0
