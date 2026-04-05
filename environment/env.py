from __future__ import annotations

import importlib
import random
import uuid
from typing import Any, Dict, Optional, Tuple

from environment.models import (
    AppointmentSlot,
    ConfirmedBookingSummary,
    PatientRequest,
    RewardBreakdown,
    SchedulerAction,
    SchedulerObservation,
    SchedulerState,
    StepResponse,
)

# ---------------------------------------------------------------------------
# Registry — maps task name to dotted class paths (lazy-loaded)
# ---------------------------------------------------------------------------

_TASK_REGISTRY: Dict[str, str] = {
    "easy":   "environment.tasks.task_easy.EasyTask",
    "medium": "environment.tasks.task_medium.MediumTask",
    "hard":   "environment.tasks.task_hard.HardTask",
}

_GRADER_REGISTRY: Dict[str, str] = {
    "easy":   "environment.graders.grader_easy.EasyGrader",
    "medium": "environment.graders.grader_medium.MediumGrader",
    "hard":   "environment.graders.grader_hard.HardGrader",
}

# Max steps per difficulty — enough room to schedule all patients with
# check/view actions, but not infinite to discourage padding.
MAX_STEPS: Dict[str, int] = {
    "easy":   10,
    "medium": 25,
    "hard":   40,
}

# Urgency priority rank (higher = more urgent)
URGENCY_RANK: Dict[str, int] = {
    "critical": 3,
    "high":     2,
    "medium":   1,
    "low":      0,
}

# Penalty constants (subtracted from step reward; always clamped ≥ 0.0)
_PENALTY_INVALID   = 0.05
_PENALTY_SPECIALTY = 0.05
_PENALTY_SLOT      = 0.05


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class HealthcareAppointmentEnv:
    """
    OpenEnv-compatible environment for healthcare appointment coordination.

    Public API:
        reset(task_name, seed)  → SchedulerObservation
        step(action)            → StepResponse
        state()                 → SchedulerState
    """

    def __init__(self) -> None:
        self._state: Optional[SchedulerState] = None
        self._task = None
        self._grader = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_name: str = "easy",
        seed: Optional[int] = None,
    ) -> SchedulerObservation:
        """
        Start a new episode.

        Loads the task class for the requested difficulty, generates a
        deterministic scenario from `seed`, initialises SchedulerState,
        and returns the initial observation.
        """
        if task_name not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task_name}'. Valid options: {list(_TASK_REGISTRY)}"
            )

        seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        self._task = self._load_class(_TASK_REGISTRY[task_name])(seed=seed)
        self._grader = self._load_class(_GRADER_REGISTRY[task_name])()

        scenario: Dict[str, Any] = self._task.generate_scenario()

        self._state = SchedulerState(
            session_id=str(uuid.uuid4()),
            task_name=task_name,
            seed=seed,
            step_count=0,
            max_steps=MAX_STEPS[task_name],
            all_requests=scenario["requests"],
            all_slots=scenario["slots"],
            all_doctors=scenario["doctors"],
        )

        n = len(self._state.all_requests)
        return self._build_observation(
            message=(
                f"Episode started | task={task_name} | seed={seed} | "
                f"patients={n} | max_steps={MAX_STEPS[task_name]}. "
                f"Schedule all {n} patient(s) correctly within the step budget."
            )
        )

    def step(self, action: SchedulerAction) -> StepResponse:
        """
        Apply one agent action and advance the environment by one step.

        Flow:
          1. Guard checks (reset called, episode not done).
          2. Increment step counter.
          3. Validate action IDs against live state.
          4. Dispatch to action handler → (result_msg, penalty, newly_scheduled).
          5. Check episode completion.
          6. Compute reward (partial progress or final grader score).
          7. Return StepResponse with observation + diagnostics.
        """
        self._assert_ready()

        self._state.step_count += 1

        valid, reason = self._validate_action(action)
        if not valid:
            obs = self._build_observation(last_action_result=f"[INVALID] {reason}")
            return StepResponse(
                observation=obs,
                reward=0.0,
                done=False,
                info={
                    "error": reason,
                    "penalty": _PENALTY_INVALID,
                    "step": self._state.step_count,
                    "steps_remaining": self._state.max_steps - self._state.step_count,
                },
            )

        result_msg, penalty, newly_scheduled = self._apply_action(action)

        done = self._check_done()
        self._state.done = done

        reward = self._compute_reward(
            done=done, penalty=penalty, newly_scheduled=newly_scheduled
        )

        obs = self._build_observation(last_action_result=result_msg, done=done)
        info = self._build_info(action=action, penalty=penalty)

        return StepResponse(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> SchedulerState:
        """
        Return a deep copy of the full internal state.
        Safe for graders and external tooling — mutations do not affect the env.
        """
        self._assert_ready()
        return self._state.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Task / grader loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_class(dotted_path: str):
        """Lazy-import a class from a dotted module.class_name path."""
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    # ------------------------------------------------------------------
    # Action validation
    # ------------------------------------------------------------------

    def _validate_action(self, action: SchedulerAction) -> Tuple[bool, str]:
        """
        Environment-level ID checks that run after Pydantic schema validation.

        Verifies that all referenced patient_ids, slot_ids, and new_slot_ids
        actually exist in the current episode's state.
        """
        s = self._state
        patient_ids = {r.patient_id for r in s.all_requests}
        slot_ids    = {sl.slot_id   for sl in s.all_slots}

        if action.patient_id and action.patient_id not in patient_ids:
            return False, f"patient_id '{action.patient_id}' does not exist in this episode."

        if action.slot_id and action.slot_id not in slot_ids:
            return False, f"slot_id '{action.slot_id}' does not exist in this episode."

        if action.new_slot_id and action.new_slot_id not in slot_ids:
            return False, f"new_slot_id '{action.new_slot_id}' does not exist in this episode."

        if action.action_type == "schedule_slot":
            slot = self._slot_by_id(action.slot_id)
            if not slot.is_available:
                return False, f"slot_id '{action.slot_id}' is already booked."
            if action.patient_id in s.scheduled:
                return False, (
                    f"Patient '{action.patient_id}' is already scheduled. "
                    "Use reschedule_slot to move the booking."
                )

        if action.action_type == "reschedule_slot":
            if action.patient_id not in s.scheduled:
                return False, (
                    f"Patient '{action.patient_id}' has no existing booking to reschedule."
                )
            new_slot = self._slot_by_id(action.new_slot_id)
            if not new_slot.is_available:
                return False, f"new_slot_id '{action.new_slot_id}' is not available."

        if action.action_type == "confirm_booking":
            existing = s.scheduled.get(action.patient_id)
            if not existing:
                return False, (
                    f"Patient '{action.patient_id}' has no scheduled booking to confirm."
                )
            if existing != action.slot_id:
                return False, (
                    f"slot_id '{action.slot_id}' does not match the patient's "
                    f"current booking '{existing}'."
                )

        return True, ""

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _apply_action(
        self, action: SchedulerAction
    ) -> Tuple[str, float, int]:
        """
        Route action_type to its handler.

        Returns: (result_message, penalty_incurred, newly_scheduled_count)
        """
        dispatch = {
            "view_request":       self._handle_view_request,
            "check_availability": self._handle_check_availability,
            "assign_specialist":  self._handle_assign_specialist,
            "schedule_slot":      self._handle_schedule_slot,
            "reschedule_slot":    self._handle_reschedule_slot,
            "flag_emergency":     self._handle_flag_emergency,
            "confirm_booking":    self._handle_confirm_booking,
        }
        return dispatch[action.action_type](action)

    def _handle_view_request(self, action: SchedulerAction) -> Tuple[str, float, int]:
        p = self._patient_by_id(action.patient_id)
        scheduled_to = self._state.scheduled.get(p.patient_id, "unscheduled")
        return (
            f"Patient {p.name} (id={p.patient_id}): condition='{p.condition}', "
            f"urgency={p.urgency}, specialty={p.required_specialty}, "
            f"preferred={p.preferred_date} [{p.preferred_time_window}], "
            f"emergency={p.is_emergency}, status={scheduled_to}",
            0.0,
            0,
        )

    def _handle_check_availability(self, action: SchedulerAction) -> Tuple[str, float, int]:
        slots = [sl for sl in self._state.all_slots if sl.is_available]
        if action.doctor_id:
            slots = [sl for sl in slots if sl.doctor_id == action.doctor_id]
        summary = [
            f"{sl.slot_id}(dr={sl.doctor_id}, {sl.date} {sl.start_time}-{sl.end_time} [{sl.time_window}])"
            for sl in slots[:15]
        ]
        suffix = f" ... +{len(slots) - 15} more" if len(slots) > 15 else ""
        return f"Available slots [{len(slots)} total]: {', '.join(summary)}{suffix}", 0.0, 0

    def _handle_assign_specialist(self, action: SchedulerAction) -> Tuple[str, float, int]:
        p = self._patient_by_id(action.patient_id)
        specialty = (action.specialty or "").strip()
        self._state.specialty_assignments[action.patient_id] = specialty

        if specialty.lower() != p.required_specialty.lower():
            return (
                f"[WARN] Assigned specialty '{specialty}' does not match "
                f"required '{p.required_specialty}' for {p.name}.",
                _PENALTY_SPECIALTY,
                0,
            )
        return (
            f"Specialty '{specialty}' correctly assigned to {p.name}.",
            0.0,
            0,
        )

    def _handle_schedule_slot(self, action: SchedulerAction) -> Tuple[str, float, int]:
        p    = self._patient_by_id(action.patient_id)
        slot = self._slot_by_id(action.slot_id)
        doc  = self._doctor_by_id(slot.doctor_id)

        penalty = 0.0
        warn = ""
        if doc.specialty.lower() != p.required_specialty.lower():
            penalty = _PENALTY_SPECIALTY
            warn = (
                f" [WARN: specialty mismatch — patient needs '{p.required_specialty}', "
                f"doctor provides '{doc.specialty}']"
            )

        slot.is_available      = False
        slot.booked_patient_id = action.patient_id
        self._state.scheduled[action.patient_id] = action.slot_id

        return (
            f"Scheduled {p.name} → Dr. {doc.name} ({doc.specialty}) "
            f"on {slot.date} {slot.start_time}–{slot.end_time}.{warn}",
            penalty,
            1,
        )

    def _handle_reschedule_slot(self, action: SchedulerAction) -> Tuple[str, float, int]:
        p        = self._patient_by_id(action.patient_id)
        old_slot = self._slot_by_id(action.slot_id)
        new_slot = self._slot_by_id(action.new_slot_id)
        doc      = self._doctor_by_id(new_slot.doctor_id)

        # Release old slot
        old_slot.is_available      = True
        old_slot.booked_patient_id = None

        # Book new slot
        new_slot.is_available      = False
        new_slot.booked_patient_id = action.patient_id
        self._state.scheduled[action.patient_id] = action.new_slot_id

        penalty = 0.0
        warn = ""
        if doc.specialty.lower() != p.required_specialty.lower():
            penalty = _PENALTY_SPECIALTY
            warn = f" [WARN: specialty mismatch '{doc.specialty}' ≠ '{p.required_specialty}']"

        return (
            f"Rescheduled {p.name} → Dr. {doc.name} "
            f"on {new_slot.date} {new_slot.start_time}–{new_slot.end_time}.{warn}",
            penalty,
            0,
        )

    def _handle_flag_emergency(self, action: SchedulerAction) -> Tuple[str, float, int]:
        p = self._patient_by_id(action.patient_id)
        if action.patient_id not in self._state.emergency_flags:
            self._state.emergency_flags.append(action.patient_id)
        # Upgrade urgency in the live request object
        for i, req in enumerate(self._state.all_requests):
            if req.patient_id == action.patient_id:
                self._state.all_requests[i] = req.model_copy(
                    update={"urgency": "critical", "is_emergency": True}
                )
                break
        return (
            f"Emergency flag set for {p.name}. Urgency upgraded to 'critical'.",
            0.0,
            0,
        )

    def _handle_confirm_booking(self, action: SchedulerAction) -> Tuple[str, float, int]:
        p    = self._patient_by_id(action.patient_id)
        slot = self._slot_by_id(action.slot_id)
        doc  = self._doctor_by_id(slot.doctor_id)
        return (
            f"Booking confirmed: {p.name} → Dr. {doc.name} "
            f"on {slot.date} at {slot.start_time}.",
            0.0,
            0,
        )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self, done: bool, penalty: float, newly_scheduled: int
    ) -> float:
        """
        Reward strategy:
        - Terminal step: full grader score (0.0–1.0).
        - Non-terminal:  small progress signal per new booking minus penalty.
          Progress bonus = (1 / total_patients) * 0.1
          This keeps intermediate rewards small relative to the final score.
        """
        if done:
            breakdown: RewardBreakdown = self._grader.grade(self._state)
            self._state.score            = breakdown.total
            self._state.reward_breakdown = breakdown
            return round(breakdown.total, 4)

        n_total = len(self._state.all_requests)
        progress_bonus = (newly_scheduled / max(n_total, 1)) * 0.1
        return round(max(0.0, progress_bonus - penalty), 4)

    # ------------------------------------------------------------------
    # Completion check
    # ------------------------------------------------------------------

    def _check_done(self) -> bool:
        """
        Episode ends when:
          (a) every patient has a confirmed booking, OR
          (b) the step budget is exhausted.
        """
        all_booked  = len(self._state.scheduled) >= len(self._state.all_requests)
        out_of_time = self._state.step_count >= self._state.max_steps
        return all_booked or out_of_time

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        last_action_result: Optional[str] = None,
        message: Optional[str] = None,
        done: bool = False,
    ) -> SchedulerObservation:
        s = self._state
        scheduled_ids = set(s.scheduled.keys())

        pending   = [r for r in s.all_requests if r.patient_id not in scheduled_ids]
        available = [sl for sl in s.all_slots   if sl.is_available]

        confirmed: list[ConfirmedBookingSummary] = []
        for pid, sid in s.scheduled.items():
            p   = self._patient_by_id(pid)
            sl  = self._slot_by_id(sid)
            doc = self._doctor_by_id(sl.doctor_id)
            confirmed.append(
                ConfirmedBookingSummary(
                    patient_id   = pid,
                    patient_name = p.name,
                    doctor_id    = sl.doctor_id,
                    doctor_name  = doc.name,
                    slot_id      = sid,
                    date         = sl.date,
                    start_time   = sl.start_time,
                    end_time     = sl.end_time,
                    specialty    = doc.specialty,
                    urgency      = p.urgency,
                )
            )

        return SchedulerObservation(
            step                = s.step_count,
            pending_requests    = pending,
            available_slots     = available,
            confirmed_bookings  = confirmed,
            last_action_result  = last_action_result,
            message             = message,
            done                = done,
        )

    # ------------------------------------------------------------------
    # Diagnostics info builder
    # ------------------------------------------------------------------

    def _build_info(
        self, action: SchedulerAction, penalty: float
    ) -> Dict[str, Any]:
        s = self._state
        n_total     = len(s.all_requests)
        n_scheduled = len(s.scheduled)
        return {
            "step":                s.step_count,
            "max_steps":           s.max_steps,
            "steps_remaining":     s.max_steps - s.step_count,
            "patients_total":      n_total,
            "patients_scheduled":  n_scheduled,
            "patients_remaining":  n_total - n_scheduled,
            "coverage_so_far":     round(n_scheduled / max(n_total, 1), 3),
            "emergency_flags":     list(s.emergency_flags),
            "action_type":         action.action_type,
            "penalty":             penalty,
        }

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if self._state is None:
            raise RuntimeError("Call reset() before interacting with the environment.")
        if self._state.done:
            raise RuntimeError("Episode is already done. Call reset() to start a new episode.")

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def _patient_by_id(self, patient_id: str) -> PatientRequest:
        for r in self._state.all_requests:
            if r.patient_id == patient_id:
                return r
        raise KeyError(f"Patient '{patient_id}' not found in this episode.")

    def _slot_by_id(self, slot_id: str) -> AppointmentSlot:
        for sl in self._state.all_slots:
            if sl.slot_id == slot_id:
                return sl
        raise KeyError(f"Slot '{slot_id}' not found in this episode.")

    def _doctor_by_id(self, doctor_id: str):
        for d in self._state.all_doctors:
            if d.doctor_id == doctor_id:
                return d
        raise KeyError(f"Doctor '{doctor_id}' not found in this episode.")
