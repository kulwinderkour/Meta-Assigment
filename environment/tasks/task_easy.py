from __future__ import annotations

import random
from typing import Any, Dict

from environment.models import AppointmentSlot, DoctorProfile, PatientRequest

# Fixed clinic dates (deterministic; seed only varies names/ages)
_DATE_1 = "2025-06-02"  # Monday
_DATE_2 = "2025-06-03"  # Tuesday

_FIRST_NAMES = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry"]
_LAST_NAMES  = ["Thompson", "Martinez", "Johnson", "Lee", "Wilson", "Davis", "Miller", "Brown"]


class EasyTask:
    """
    Easy task: 1 patient, 3 doctors (only 1 correct specialty), 6 slots.

    Scenario
    --------
    A single patient presents with a persistent headache and visual
    disturbances and needs a neurology appointment on Monday morning.

    Three doctors are available:
      d001 – Dr. Sarah Chen      (neurology)   ← correct specialty
      d002 – Dr. James Liu       (dermatology) ← wrong specialty
      d003 – Dr. Patricia Brown  (orthopedics) ← wrong specialty

    Six slots exist across both days and time windows.  The optimal
    booking is s001 (d001 / neurology / DATE_1 / morning) because it
    matches both the required specialty AND the patient's preferred
    date + time window.  s002 is a partial match (correct specialty,
    wrong time window).  All other slots are wrong specialty.

    Ground truth
    ------------
    optimal_slot     : "s001"
    acceptable_slots : ["s001", "s002"]   (neurology, any window)
    emergency_flags  : []
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def generate_scenario(self) -> Dict[str, Any]:
        rng = self._rng
        name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        age  = rng.randint(28, 72)

        doctors = [
            DoctorProfile(
                doctor_id="d001",
                name="Dr. Sarah Chen",
                specialty="neurology",
                available_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                available_time_windows=["morning", "afternoon"],
                max_patients_per_day=8,
            ),
            DoctorProfile(
                doctor_id="d002",
                name="Dr. James Liu",
                specialty="dermatology",
                available_days=["Monday", "Tuesday", "Wednesday"],
                available_time_windows=["morning"],
                max_patients_per_day=6,
            ),
            DoctorProfile(
                doctor_id="d003",
                name="Dr. Patricia Brown",
                specialty="orthopedics",
                available_days=["Monday", "Wednesday", "Friday"],
                available_time_windows=["morning", "afternoon"],
                max_patients_per_day=6,
            ),
        ]

        slots = [
            # s001 — OPTIMAL: correct specialty + preferred date + preferred window
            AppointmentSlot(slot_id="s001", doctor_id="d001", date=_DATE_1,
                            start_time="09:00", end_time="09:30", time_window="morning"),
            # s002 — ACCEPTABLE: correct specialty, wrong time window
            AppointmentSlot(slot_id="s002", doctor_id="d001", date=_DATE_1,
                            start_time="14:00", end_time="14:30", time_window="afternoon"),
            # s003–s004 — WRONG: dermatology
            AppointmentSlot(slot_id="s003", doctor_id="d002", date=_DATE_1,
                            start_time="09:30", end_time="10:00", time_window="morning"),
            AppointmentSlot(slot_id="s004", doctor_id="d002", date=_DATE_2,
                            start_time="10:00", end_time="10:30", time_window="morning"),
            # s005–s006 — WRONG: orthopedics
            AppointmentSlot(slot_id="s005", doctor_id="d003", date=_DATE_1,
                            start_time="09:00", end_time="09:30", time_window="morning"),
            AppointmentSlot(slot_id="s006", doctor_id="d003", date=_DATE_1,
                            start_time="15:00", end_time="15:30", time_window="afternoon"),
        ]

        patients = [
            PatientRequest(
                patient_id="p001",
                name=name,
                age=age,
                condition="persistent headache with visual disturbances",
                urgency="medium",
                preferred_date=_DATE_1,
                preferred_time_window="morning",
                required_specialty="neurology",
                notes="2-week history of headaches. Prefers Monday morning appointment.",
            ),
        ]

        ground_truth = {
            "optimal_slot":      "s001",
            "acceptable_slots":  ["s001", "s002"],
            "emergency_flags":   [],
            "correct_specialties": {"p001": "neurology"},
        }

        return {
            "requests":     patients,
            "slots":        slots,
            "doctors":      doctors,
            "ground_truth": ground_truth,
        }
