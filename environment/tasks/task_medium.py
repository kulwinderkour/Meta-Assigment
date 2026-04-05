from __future__ import annotations

import random
from typing import Any, Dict

from environment.models import AppointmentSlot, DoctorProfile, PatientRequest

_DATE_1 = "2025-06-02"  # Monday
_DATE_2 = "2025-06-03"  # Tuesday

_FIRST_NAMES = ["James", "Maria", "Robert", "Linda", "William", "Susan", "Kevin", "Anna"]
_LAST_NAMES  = ["Harris", "Clark", "Lewis", "Robinson", "Walker", "Hall", "Young", "King"]


class MediumTask:
    """
    Medium task: 4 patients, 4 doctors, 12 slots.

    Scenario
    --------
    Four patients arrive simultaneously with different specialties and
    urgency levels.  The cardiologist (d001) has only 2 morning slots
    available — a scarce resource.

    Doctors:
      d001 – Dr. Michael Torres  (cardiology)       mornings only, Mon/Wed
      d002 – Dr. Jennifer Park   (orthopedics)      Mon/Tue/Thu, any window
      d003 – Dr. Carlos Reyes    (dermatology)      Mon/Tue/Wed, any window
      d004 – Dr. Linda Walsh     (general practice) Mon–Fri, any window

    Patients and urgency:
      p001 – critical  – cardiology     ← must get first available cardiology slot
      p002 – high      – orthopedics
      p003 – medium    – dermatology    (prefers afternoon)
      p004 – low       – general practice (prefers afternoon)

    Key challenge
    -------------
    Cardiology slots are scarce (s001 and s002 only).  The agent must
    give the critical patient (p001) priority — i.e., book p001 into
    s001 (the earliest cardiology slot).  All 4 patients must end up
    with the correct specialty.

    Ground truth
    ------------
    correct_specialties  : per-patient specialty mapping
    urgency_order        : ["p001", "p002", "p003", "p004"]
    cardiology_priority  : "p001" must occupy the earliest cardiology slot
    emergency_flags      : []
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._rng = random.Random(seed)

    def generate_scenario(self) -> Dict[str, Any]:
        rng = self._rng

        def _name() -> str:
            return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"

        doctors = [
            DoctorProfile(
                doctor_id="d001",
                name="Dr. Michael Torres",
                specialty="cardiology",
                available_days=["Monday", "Wednesday"],
                available_time_windows=["morning"],
                max_patients_per_day=4,
            ),
            DoctorProfile(
                doctor_id="d002",
                name="Dr. Jennifer Park",
                specialty="orthopedics",
                available_days=["Monday", "Tuesday", "Thursday"],
                available_time_windows=["morning", "afternoon"],
                max_patients_per_day=6,
            ),
            DoctorProfile(
                doctor_id="d003",
                name="Dr. Carlos Reyes",
                specialty="dermatology",
                available_days=["Monday", "Tuesday", "Wednesday"],
                available_time_windows=["morning", "afternoon"],
                max_patients_per_day=8,
            ),
            DoctorProfile(
                doctor_id="d004",
                name="Dr. Linda Walsh",
                specialty="general practice",
                available_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                available_time_windows=["morning", "afternoon", "evening"],
                max_patients_per_day=12,
            ),
        ]

        slots = [
            # Cardiology — only 2 morning slots (scarce)
            AppointmentSlot(slot_id="s001", doctor_id="d001", date=_DATE_1,
                            start_time="08:30", end_time="09:00", time_window="morning"),
            AppointmentSlot(slot_id="s002", doctor_id="d001", date=_DATE_1,
                            start_time="09:00", end_time="09:30", time_window="morning"),
            # Orthopedics — 3 slots
            AppointmentSlot(slot_id="s003", doctor_id="d002", date=_DATE_1,
                            start_time="09:00", end_time="09:30", time_window="morning"),
            AppointmentSlot(slot_id="s004", doctor_id="d002", date=_DATE_1,
                            start_time="14:00", end_time="14:30", time_window="afternoon"),
            AppointmentSlot(slot_id="s005", doctor_id="d002", date=_DATE_2,
                            start_time="09:00", end_time="09:30", time_window="morning"),
            # Dermatology — 3 slots
            AppointmentSlot(slot_id="s006", doctor_id="d003", date=_DATE_1,
                            start_time="10:00", end_time="10:30", time_window="morning"),
            AppointmentSlot(slot_id="s007", doctor_id="d003", date=_DATE_1,
                            start_time="14:30", end_time="15:00", time_window="afternoon"),
            AppointmentSlot(slot_id="s008", doctor_id="d003", date=_DATE_2,
                            start_time="10:00", end_time="10:30", time_window="morning"),
            # General Practice — 4 slots
            AppointmentSlot(slot_id="s009", doctor_id="d004", date=_DATE_1,
                            start_time="09:30", end_time="10:00", time_window="morning"),
            AppointmentSlot(slot_id="s010", doctor_id="d004", date=_DATE_1,
                            start_time="15:00", end_time="15:30", time_window="afternoon"),
            AppointmentSlot(slot_id="s011", doctor_id="d004", date=_DATE_1,
                            start_time="17:30", end_time="18:00", time_window="evening"),
            AppointmentSlot(slot_id="s012", doctor_id="d004", date=_DATE_2,
                            start_time="09:00", end_time="09:30", time_window="morning"),
        ]

        patients = [
            PatientRequest(
                patient_id="p001",
                name=_name(),
                age=rng.randint(55, 75),
                condition="acute chest pain with shortness of breath",
                urgency="critical",
                preferred_date=_DATE_1,
                preferred_time_window="morning",
                required_specialty="cardiology",
                notes="Referred from A&E. Needs immediate cardiac evaluation today.",
            ),
            PatientRequest(
                patient_id="p002",
                name=_name(),
                age=rng.randint(30, 55),
                condition="knee injury from sports accident",
                urgency="high",
                preferred_date=_DATE_1,
                preferred_time_window="morning",
                required_specialty="orthopedics",
                notes="Cannot fully bear weight. X-ray required.",
            ),
            PatientRequest(
                patient_id="p003",
                name=_name(),
                age=rng.randint(20, 45),
                condition="recurring skin rash on arms",
                urgency="medium",
                preferred_date=_DATE_1,
                preferred_time_window="afternoon",
                required_specialty="dermatology",
                notes="Rash persisting for 3 weeks. Flexible scheduling.",
            ),
            PatientRequest(
                patient_id="p004",
                name=_name(),
                age=rng.randint(18, 40),
                condition="general fatigue and mild cold symptoms",
                urgency="low",
                preferred_date=_DATE_1,
                preferred_time_window="afternoon",
                required_specialty="general practice",
                notes="Routine consultation. Flexible timing.",
            ),
        ]

        ground_truth = {
            "correct_specialties": {
                "p001": "cardiology",
                "p002": "orthopedics",
                "p003": "dermatology",
                "p004": "general practice",
            },
            # Expected urgency ordering (descending) — used by grader for priority check
            "urgency_order":      ["p001", "p002", "p003", "p004"],
            # The critical patient must occupy the earliest cardiology slot (s001)
            "cardiology_priority": "p001",
            "emergency_flags":    [],
        }

        return {
            "requests":     patients,
            "slots":        slots,
            "doctors":      doctors,
            "ground_truth": ground_truth,
        }
