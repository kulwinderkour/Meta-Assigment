from __future__ import annotations

import random
from typing import Any, Dict

from environment.models import AppointmentSlot, DoctorProfile, PatientRequest

_DATE_1 = "2025-06-02"  # Monday
_DATE_2 = "2025-06-03"  # Tuesday

_FIRST_NAMES = ["Thomas", "Olivia", "Noah", "Sophia", "Ethan", "Isabella", "Mason", "Mia"]
_LAST_NAMES  = ["Anderson", "Jackson", "White", "Harris", "Martin", "Garcia", "Rodriguez", "Wilson"]


class HardTask:
    """
    Hard task: 8 patients, 5 doctors, 20 slots across 2 days.

    Scenario
    --------
    Eight patients arrive at the clinic with mixed urgency levels and
    specialties.  Specialist slots are scarce; the agent must manage
    priority, detect an undeclared emergency, and potentially reschedule.

    Doctors:
      d001 – Dr. Victor Mendes   (cardiology)       mornings only — 3 slots total
      d002 – Dr. Rachel Kim      (neurology)        2 mornings + 1 afternoon — 3 slots total
      d003 – Dr. Ahmed Hassan    (orthopedics)      Mon+Tue, any window — 4 slots
      d004 – Dr. Fiona Gallagher (dermatology)      Mon+Thu, any window — 3 slots
      d005 – Dr. Sean O'Brien    (general practice) Mon–Fri, any window — 7 slots

    Patients:
      p001 – critical – cardiology   (needs earliest cardiology slot)
      p002 – high     – cardiology   (must yield to p001)
      p003 – medium   – neurology
      p004 – high     – neurology    ← MUST BE FLAGGED AS EMERGENCY before scheduling
                                       (notes describe thunderclap headache / SAH risk)
      p005 – medium   – orthopedics
      p006 – medium   – dermatology
      p007 – low      – general practice (notes: mild palpitations, GP is sufficient —
                                           do NOT route to cardiology)
      p008 – low      – general practice (routine, flexible)

    Key challenges
    --------------
    1. Emergency escalation: p004 urgency starts as "high" but the clinical
       notes explicitly require EMERGENCY ESCALATION.  Agent must call
       flag_emergency(p004) which upgrades urgency to "critical" so that
       p004 gets the earlier neurology slot (s004) over p003 (medium).

    2. Rescheduling trap: if the agent greedily books p002 into s001
       (earliest cardiology slot) before recognising that p001 is critical,
       it must later reschedule_slot p002 to s002 and assign p001 to s001.

    3. Specialty routing: p007 mentions palpitations — a naive agent might
       try to send it to cardiology.  The notes explicitly say GP only.
       Cardiology slots are scarce and must not be wasted.

    4. All 8 patients must be scheduled across 2 days with no conflicts.

    Ground truth
    ------------
    correct_specialties : per-patient mapping
    emergency_flags     : ["p004"]   — p004 MUST be flagged before being scheduled
    urgency_priority_pairs: same-specialty pairs where order matters
      ("p001", "p002") — cardiology: critical > high → p001 in s001, p002 in s002
      ("p004", "p003") — neurology:  after flag critical > medium → p004 in s004, p003 in s005
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
                name="Dr. Victor Mendes",
                specialty="cardiology",
                available_days=["Monday", "Tuesday"],
                available_time_windows=["morning"],
                max_patients_per_day=4,
            ),
            DoctorProfile(
                doctor_id="d002",
                name="Dr. Rachel Kim",
                specialty="neurology",
                available_days=["Monday", "Tuesday"],
                available_time_windows=["morning", "afternoon"],
                max_patients_per_day=4,
            ),
            DoctorProfile(
                doctor_id="d003",
                name="Dr. Ahmed Hassan",
                specialty="orthopedics",
                available_days=["Monday", "Tuesday", "Thursday"],
                available_time_windows=["morning", "afternoon"],
                max_patients_per_day=6,
            ),
            DoctorProfile(
                doctor_id="d004",
                name="Dr. Fiona Gallagher",
                specialty="dermatology",
                available_days=["Monday", "Wednesday", "Thursday"],
                available_time_windows=["morning", "afternoon"],
                max_patients_per_day=8,
            ),
            DoctorProfile(
                doctor_id="d005",
                name="Dr. Sean O'Brien",
                specialty="general practice",
                available_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                available_time_windows=["morning", "afternoon", "evening"],
                max_patients_per_day=12,
            ),
        ]

        slots = [
            # --- Cardiology: 3 morning slots (d001) ---
            # s001 = earliest; p001 (critical) must occupy this
            AppointmentSlot(slot_id="s001", doctor_id="d001", date=_DATE_1,
                            start_time="08:30", end_time="09:00", time_window="morning"),
            # s002 = second; p002 (high) goes here
            AppointmentSlot(slot_id="s002", doctor_id="d001", date=_DATE_1,
                            start_time="09:00", end_time="09:30", time_window="morning"),
            # s003 = spill-over on Tuesday (not needed if urgency order correct)
            AppointmentSlot(slot_id="s003", doctor_id="d001", date=_DATE_2,
                            start_time="08:30", end_time="09:00", time_window="morning"),

            # --- Neurology: 2 morning + 1 afternoon (d002) ---
            # s004 = earliest; p004 (after flag → critical) must occupy this
            AppointmentSlot(slot_id="s004", doctor_id="d002", date=_DATE_1,
                            start_time="09:00", end_time="09:30", time_window="morning"),
            # s005 = second morning; p003 (medium) goes here
            AppointmentSlot(slot_id="s005", doctor_id="d002", date=_DATE_1,
                            start_time="09:30", end_time="10:00", time_window="morning"),
            # s006 = afternoon spill-over
            AppointmentSlot(slot_id="s006", doctor_id="d002", date=_DATE_1,
                            start_time="14:00", end_time="14:30", time_window="afternoon"),

            # --- Orthopedics: 4 slots (d003) ---
            AppointmentSlot(slot_id="s007", doctor_id="d003", date=_DATE_1,
                            start_time="09:00", end_time="09:30", time_window="morning"),
            AppointmentSlot(slot_id="s008", doctor_id="d003", date=_DATE_1,
                            start_time="14:00", end_time="14:30", time_window="afternoon"),
            AppointmentSlot(slot_id="s009", doctor_id="d003", date=_DATE_2,
                            start_time="09:00", end_time="09:30", time_window="morning"),
            AppointmentSlot(slot_id="s010", doctor_id="d003", date=_DATE_2,
                            start_time="14:00", end_time="14:30", time_window="afternoon"),

            # --- Dermatology: 3 slots (d004) ---
            AppointmentSlot(slot_id="s011", doctor_id="d004", date=_DATE_1,
                            start_time="10:00", end_time="10:30", time_window="morning"),
            AppointmentSlot(slot_id="s012", doctor_id="d004", date=_DATE_1,
                            start_time="14:30", end_time="15:00", time_window="afternoon"),
            AppointmentSlot(slot_id="s013", doctor_id="d004", date=_DATE_2,
                            start_time="10:00", end_time="10:30", time_window="morning"),

            # --- General Practice: 7 slots (d005) ---
            AppointmentSlot(slot_id="s014", doctor_id="d005", date=_DATE_1,
                            start_time="08:30", end_time="09:00", time_window="morning"),
            AppointmentSlot(slot_id="s015", doctor_id="d005", date=_DATE_1,
                            start_time="10:00", end_time="10:30", time_window="morning"),
            AppointmentSlot(slot_id="s016", doctor_id="d005", date=_DATE_1,
                            start_time="14:00", end_time="14:30", time_window="afternoon"),
            AppointmentSlot(slot_id="s017", doctor_id="d005", date=_DATE_1,
                            start_time="17:30", end_time="18:00", time_window="evening"),
            AppointmentSlot(slot_id="s018", doctor_id="d005", date=_DATE_2,
                            start_time="09:00", end_time="09:30", time_window="morning"),
            AppointmentSlot(slot_id="s019", doctor_id="d005", date=_DATE_2,
                            start_time="14:00", end_time="14:30", time_window="afternoon"),
            AppointmentSlot(slot_id="s020", doctor_id="d005", date=_DATE_2,
                            start_time="17:30", end_time="18:00", time_window="evening"),
        ]

        patients = [
            # --- Cardiology pair ---
            PatientRequest(
                patient_id="p001",
                name=_name(),
                age=rng.randint(60, 80),
                condition="acute myocardial infarction symptoms",
                urgency="critical",
                preferred_date=_DATE_1,
                preferred_time_window="morning",
                required_specialty="cardiology",
                notes=(
                    "Chest pain radiating to left arm. ECG shows ST elevation. "
                    "Immediate cardiac review required. Highest priority."
                ),
            ),
            PatientRequest(
                patient_id="p002",
                name=_name(),
                age=rng.randint(45, 65),
                condition="stable angina with recent escalation",
                urgency="high",
                preferred_date=_DATE_1,
                preferred_time_window="morning",
                required_specialty="cardiology",
                notes="Cardiac follow-up needed within 24 hours.",
            ),
            # --- Neurology pair (p004 is the undeclared emergency) ---
            PatientRequest(
                patient_id="p003",
                name=_name(),
                age=rng.randint(25, 50),
                condition="cluster migraines recurring weekly",
                urgency="medium",
                preferred_date=_DATE_1,
                preferred_time_window="morning",
                required_specialty="neurology",
                notes="Migraines disrupting daily work. Non-emergency; routine neurology review.",
            ),
            PatientRequest(
                patient_id="p004",
                name=_name(),
                age=rng.randint(30, 55),
                condition="sudden onset severe headache — worst of patient's life",
                urgency="high",   # Must be upgraded to critical via flag_emergency
                preferred_date=_DATE_1,
                preferred_time_window="morning",
                required_specialty="neurology",
                notes=(
                    "Thunderclap headache with sudden onset and neck stiffness. "
                    "Rule out subarachnoid haemorrhage. "
                    "REQUIRES EMERGENCY ESCALATION — flag this patient immediately."
                ),
            ),
            # --- Orthopedics ---
            PatientRequest(
                patient_id="p005",
                name=_name(),
                age=rng.randint(20, 50),
                condition="suspected fractured wrist after fall",
                urgency="medium",
                preferred_date=_DATE_1,
                preferred_time_window="morning",
                required_specialty="orthopedics",
                notes="Significant wrist swelling post-fall. X-ray required.",
            ),
            # --- Dermatology ---
            PatientRequest(
                patient_id="p006",
                name=_name(),
                age=rng.randint(18, 45),
                condition="extensive psoriasis flare covering both arms and legs",
                urgency="medium",
                preferred_date=_DATE_1,
                preferred_time_window="afternoon",
                required_specialty="dermatology",
                notes="Significant coverage. Needs topical treatment plan review.",
            ),
            # --- General Practice (specialty-routing trap) ---
            PatientRequest(
                patient_id="p007",
                name=_name(),
                age=rng.randint(25, 40),
                condition="mild fatigue and occasional palpitations",
                urgency="low",
                preferred_date=_DATE_1,
                preferred_time_window="afternoon",
                required_specialty="general practice",
                notes=(
                    "Symptoms mild and non-specific. Basic bloods ordered. "
                    "GP consultation is sufficient — NOT a cardiac referral. "
                    "Do not assign to cardiology."
                ),
            ),
            # --- General Practice (routine) ---
            PatientRequest(
                patient_id="p008",
                name=_name(),
                age=rng.randint(18, 35),
                condition="annual health check and vaccination update",
                urgency="low",
                preferred_date=_DATE_2,
                preferred_time_window="morning",
                required_specialty="general practice",
                notes="Routine. Flexible timing.",
            ),
        ]

        ground_truth = {
            "correct_specialties": {
                "p001": "cardiology",
                "p002": "cardiology",
                "p003": "neurology",
                "p004": "neurology",
                "p005": "orthopedics",
                "p006": "dermatology",
                "p007": "general practice",
                "p008": "general practice",
            },
            # p004 MUST be in emergency_flags before being scheduled
            "emergency_flags": ["p004"],
            # Within same specialty, higher-urgency patient must hold the earlier slot
            # (smaller start_time on same date, or earlier date)
            "urgency_priority_pairs": [
                ("p001", "p002"),  # cardiology: critical > high
                ("p004", "p003"),  # neurology:  critical (after flag) > medium
            ],
        }

        return {
            "requests":     patients,
            "slots":        slots,
            "doctors":      doctors,
            "ground_truth": ground_truth,
        }
