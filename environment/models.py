from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations (as Literal types for lightweight determinism)
# ---------------------------------------------------------------------------

UrgencyLevel = Literal["low", "medium", "high", "critical"]
TimeWindow = Literal["morning", "afternoon", "evening"]
ActionType = Literal[
    "view_request",
    "check_availability",
    "assign_specialist",
    "schedule_slot",
    "reschedule_slot",
    "flag_emergency",
    "confirm_booking",
]
TaskDifficulty = Literal["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------

class PatientRequest(BaseModel):
    """A patient's appointment request submitted to the scheduler."""

    patient_id: str = Field(..., description="Unique patient identifier")
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=150)
    condition: str = Field(..., description="Presenting medical condition")
    urgency: UrgencyLevel
    preferred_date: str = Field(..., description="ISO date string YYYY-MM-DD")
    preferred_time_window: TimeWindow
    required_specialty: str = Field(..., description="Medical specialty needed")
    notes: Optional[str] = None
    is_emergency: bool = False

    @field_validator("preferred_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("preferred_date must be in YYYY-MM-DD format")
        return v


class AppointmentSlot(BaseModel):
    """A bookable time slot on a doctor's calendar."""

    slot_id: str = Field(..., description="Unique slot identifier")
    doctor_id: str
    date: str = Field(..., description="ISO date string YYYY-MM-DD")
    start_time: str = Field(..., description="HH:MM 24-hour format")
    end_time: str = Field(..., description="HH:MM 24-hour format")
    time_window: TimeWindow
    is_available: bool = True
    booked_patient_id: Optional[str] = None

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        if not re.fullmatch(r"\d{2}:\d{2}", v):
            raise ValueError("Time must be in HH:MM format")
        return v

    @model_validator(mode="after")
    def end_after_start(self) -> "AppointmentSlot":
        if self.start_time >= self.end_time:
            raise ValueError("end_time must be after start_time")
        return self


class DoctorProfile(BaseModel):
    """A clinic doctor with specialty and availability information."""

    doctor_id: str
    name: str = Field(..., min_length=1)
    specialty: str
    available_days: List[str] = Field(
        ..., description="Weekday names the doctor works, e.g. ['Monday', 'Wednesday']"
    )
    available_time_windows: List[TimeWindow] = Field(
        ..., description="Time windows the doctor is available in"
    )
    max_patients_per_day: int = Field(..., ge=1, le=50)

    @field_validator("available_days")
    @classmethod
    def validate_days(cls, v: List[str]) -> List[str]:
        valid = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        for day in v:
            if day not in valid:
                raise ValueError(f"Invalid day: {day}")
        return v


# ---------------------------------------------------------------------------
# Agent action model
# ---------------------------------------------------------------------------

class SchedulerAction(BaseModel):
    """
    An action submitted by the agent during a step() call.

    Supported action_types:
    - view_request        : inspect a patient's full request details
    - check_availability  : query open slots for a doctor/date/window
    - assign_specialist   : propose a specialty match for a patient
    - schedule_slot       : book a slot for a patient
    - reschedule_slot     : move an existing booking to a different slot
    - flag_emergency      : mark a patient as emergency (elevates urgency)
    - confirm_booking     : finalise and lock a confirmed appointment
    """

    action_type: ActionType
    patient_id: Optional[str] = Field(None, description="Target patient for the action")
    doctor_id: Optional[str] = Field(None, description="Target doctor for the action")
    slot_id: Optional[str] = Field(None, description="Current slot — used by schedule_slot, reschedule_slot, confirm_booking")
    new_slot_id: Optional[str] = Field(None, description="Destination slot for reschedule_slot")
    specialty: Optional[str] = Field(None, description="Used with assign_specialist")
    reason: Optional[str] = Field(None, description="Optional rationale for the action")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Extra key-value data for extensibility"
    )

    @model_validator(mode="after")
    def check_required_fields_per_action(self) -> "SchedulerAction":
        needs_patient = {
            "view_request", "assign_specialist", "flag_emergency",
            "schedule_slot", "reschedule_slot", "confirm_booking",
        }
        needs_slot = {"schedule_slot", "confirm_booking"}
        needs_both_slots = {"reschedule_slot"}

        if self.action_type in needs_patient and not self.patient_id:
            raise ValueError(f"action_type '{self.action_type}' requires patient_id")
        if self.action_type in needs_slot and not self.slot_id:
            raise ValueError(f"action_type '{self.action_type}' requires slot_id")
        if self.action_type in needs_both_slots and not (self.slot_id and self.new_slot_id):
            raise ValueError("reschedule_slot requires both slot_id (current) and new_slot_id (destination)")
        return self


# ---------------------------------------------------------------------------
# Observation model (agent-visible only)
# ---------------------------------------------------------------------------

class ConfirmedBookingSummary(BaseModel):
    """Lightweight booking summary visible to the agent."""

    patient_id: str
    patient_name: str
    doctor_id: str
    doctor_name: str
    slot_id: str
    date: str
    start_time: str
    end_time: str
    specialty: str
    urgency: UrgencyLevel


class SchedulerObservation(BaseModel):
    """
    What the agent sees after reset() or each step().

    Deliberately limited — does not expose internal grading state or
    ground-truth solution.
    """

    step: int = Field(..., ge=0)
    pending_requests: List[PatientRequest] = Field(
        ..., description="Patients not yet fully scheduled"
    )
    available_slots: List[AppointmentSlot] = Field(
        ..., description="Only slots with is_available=True"
    )
    confirmed_bookings: List[ConfirmedBookingSummary] = Field(
        default_factory=list, description="Bookings locked so far this episode"
    )
    last_action_result: Optional[str] = Field(
        None, description="Human-readable result of the most recent action"
    )
    message: Optional[str] = Field(
        None, description="Environment message or instruction hint"
    )
    done: bool = False

    @field_validator("available_slots")
    @classmethod
    def only_available_slots(cls, v: List[AppointmentSlot]) -> List[AppointmentSlot]:
        return [s for s in v if s.is_available]


# ---------------------------------------------------------------------------
# Full internal state (used by env.py and graders)
# ---------------------------------------------------------------------------

class SchedulerState(BaseModel):
    """
    Complete internal environment state.

    Not exposed directly to the agent — returned by the state() API
    for debugging, grading, and evaluation tooling.
    """

    session_id: str
    task_name: TaskDifficulty
    seed: int
    step_count: int = Field(0, ge=0)
    max_steps: int = Field(..., ge=1)

    all_requests: List[PatientRequest] = Field(
        ..., description="All patient requests for this episode"
    )
    all_slots: List[AppointmentSlot] = Field(
        ..., description="Full slot pool including booked slots"
    )
    all_doctors: List[DoctorProfile] = Field(
        ..., description="All doctors active in this episode"
    )

    scheduled: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of patient_id -> slot_id for confirmed bookings",
    )
    specialty_assignments: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of patient_id -> assigned specialty",
    )
    emergency_flags: List[str] = Field(
        default_factory=list,
        description="patient_ids that have been flagged as emergencies",
    )

    done: bool = False
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    reward_breakdown: Optional["RewardBreakdown"] = None


# ---------------------------------------------------------------------------
# Reward and step response models
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """
    Fine-grained breakdown of the final score returned by a grader.

    All component scores are normalised to [0.0, 1.0]; total is a
    weighted sum also clamped to [0.0, 1.0].
    """

    total: float = Field(..., ge=0.0, le=1.0)
    coverage: float = Field(..., ge=0.0, le=1.0, description="Fraction of patients scheduled")
    urgency_order: float = Field(
        ..., ge=0.0, le=1.0, description="Urgency priority respected"
    )
    specialty_match: float = Field(
        ..., ge=0.0, le=1.0, description="Correct specialty assigned"
    )
    no_conflicts: float = Field(
        ..., ge=0.0, le=1.0, description="No double-bookings or overlaps"
    )
    time_constraints: float = Field(
        ..., ge=0.0, le=1.0, description="Bookings within allowed time windows"
    )


class StepResponse(BaseModel):
    """HTTP response body returned by POST /step."""

    observation: SchedulerObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
