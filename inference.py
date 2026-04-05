"""
inference.py — Root-level inference runner for healthcare-appointment-coordination-env.

Runs all 3 tasks (easy → medium → hard) with a deterministic heuristic
baseline by default, while preserving optional LLM and auto-fallback modes.

Environment variables are only required for `--mode llm` or `--mode auto`:
    API_BASE_URL   Base URL for the OpenAI-compatible API endpoint
    MODEL_NAME     Model identifier (e.g. "meta-llama/Llama-3.3-70B-Instruct")
    HF_TOKEN       API key / HF token used as the bearer credential

Runtime target: < 20 minutes on a 2-vCPU / 8 GB CPU-only container.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from environment.env import HealthcareAppointmentEnv
from environment.models import SchedulerAction, SchedulerObservation

# ---------------------------------------------------------------------------
# Configuration — all sourced from env vars or compile-time constants
# ---------------------------------------------------------------------------
load_dotenv()

API_BASE_URL: str = os.getenv("API_BASE_URL", "")
MODEL_NAME:   str = os.getenv("MODEL_NAME", "")
HF_TOKEN:     str = os.getenv("HF_TOKEN", "")

# Deterministic task order and fixed seeds — same run = same result
TASK_ORDER: List[str]        = ["easy", "medium", "hard"]
TASK_SEEDS: Dict[str, int]   = {"easy": 42, "medium": 42, "hard": 42}

# LLM call guards
MAX_RETRIES:          int   = 3
RETRY_BASE_DELAY:     float = 2.0   # seconds; doubles each attempt (1× → 2× → 4×)
LLM_TIMEOUT_SECONDS:  int   = 45    # per-request timeout passed to OpenAI client

# Episode guards
MAX_PARSE_FAILS:  int = 5    # consecutive JSON parse failures before task abort
MAX_EPISODE_STEPS: int = 60  # hard cap per task (env enforces its own, this is a backup)

# Global wall-clock guard — leave 2-min buffer inside the 20-min limit
GLOBAL_TIMEOUT_SECS: int = 1080  # 18 minutes

DEFAULT_MODE: str = "heuristic"
SUPPORTED_MODES = {"heuristic", "llm", "auto"}


# ---------------------------------------------------------------------------
# Deterministic keyword rules for the heuristic agent
# ---------------------------------------------------------------------------

_SPECIALTY_RULES: List[tuple[str, tuple[str, ...]]] = [
    (
        "general practice",
        (
            "not a cardiac referral",
            "do not assign to cardiology",
            "gp consultation is sufficient",
            "general practice",
            "gp only",
            "routine",
            "annual health check",
            "vaccination",
            "mild fatigue",
            "non-specific",
            "flexible timing",
            "basic bloods ordered",
        ),
    ),
    (
        "cardiology",
        (
            "chest pain",
            "shortness of breath",
            "myocardial",
            "st elevation",
            "ecg",
            "angina",
            "cardiac",
            "heart",
            "palpitations",
        ),
    ),
    (
        "neurology",
        (
            "thunderclap",
            "subarachnoid",
            "worst of the patient's life",
            "severe headache",
            "headache",
            "migraines",
            "migraine",
            "visual disturbances",
            "visual disturbance",
            "neck stiffness",
            "neurology",
        ),
    ),
    (
        "orthopedics",
        (
            "fracture",
            "fractured",
            "knee",
            "wrist",
            "x-ray",
            "xray",
            "bone",
            "sprain",
            "fall",
            "joint",
        ),
    ),
    (
        "dermatology",
        (
            "psoriasis",
            "rash",
            "skin",
            "eczema",
            "lesion",
            "itch",
            "dermatology",
        ),
    ),
]

_EMERGENCY_HINTS: tuple[str, ...] = (
    "requires emergency escalation",
    "flag this patient immediately",
    "rule out subarachnoid haemorrhage",
    "rule out subarachnoid hemorrhage",
    "thunderclap headache",
    "worst of the patient's life",
    "sudden onset severe headache",
)

_URGENCY_RANK: Dict[str, int] = {
    "critical": 3,
    "high": 2,
    "medium": 1,
    "low": 0,
}


# ---------------------------------------------------------------------------
# System prompt (constant; sent as the "system" message on every LLM call)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a healthcare clinic appointment scheduling agent.

Goal: correctly schedule every pending patient by issuing one action at a time.

Rules:
1. Each patient has a required_specialty — only book them with a doctor of that specialty.
2. Prioritise higher urgency patients (critical > high > medium > low) when slots are scarce.
3. If a patient's notes say "REQUIRES EMERGENCY ESCALATION", call flag_emergency first.
4. Do NOT route a patient to cardiology unless their required_specialty is "cardiology".
5. Every patient must eventually receive a schedule_slot or reschedule_slot call.

Available action types and their exact required JSON fields:

  view_request        → {"action_type":"view_request","patient_id":"<id>"}
  check_availability  → {"action_type":"check_availability"}
                        (add "doctor_id":"<id>" to filter by one doctor)
  assign_specialist   → {"action_type":"assign_specialist","patient_id":"<id>","specialty":"<specialty>"}
  schedule_slot       → {"action_type":"schedule_slot","patient_id":"<id>","slot_id":"<slot_id>"}
  reschedule_slot     → {"action_type":"reschedule_slot","patient_id":"<id>",\
"slot_id":"<current_slot_id>","new_slot_id":"<destination_slot_id>"}
  flag_emergency      → {"action_type":"flag_emergency","patient_id":"<id>"}
  confirm_booking     → {"action_type":"confirm_booking","patient_id":"<id>","slot_id":"<slot_id>"}

CRITICAL OUTPUT RULE:
Respond with ONLY a single valid JSON object — no explanation, no markdown fences, no extra text.
"""


# ---------------------------------------------------------------------------
# Helper: build a structured observation prompt for the LLM
# ---------------------------------------------------------------------------

def build_observation_prompt(
    obs: SchedulerObservation,
    doctor_specialties: Dict[str, str],
) -> str:
    """
    Convert a SchedulerObservation into a structured text prompt the LLM
    can reason over and act on.

    `doctor_specialties` maps doctor_id → specialty; it is built once after
    reset() from env.state().all_doctors and remains stable for the episode.
    Including specialty in each slot line means the model never has to guess
    which doctor handles which specialty.
    """
    _urgency_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    # --- Pending patients (sorted by urgency, most urgent first) ---
    if obs.pending_requests:
        pending_lines: List[str] = []
        for p in sorted(obs.pending_requests,
                        key=lambda x: _urgency_rank.get(x.urgency, 9)):
            note = f"\n      notes: {p.notes[:140]}" if p.notes else ""
            pending_lines.append(
                f"  {p.patient_id}: {p.name}"
                f" | urgency={p.urgency}"
                f" | required_specialty={p.required_specialty}"
                f" | condition={p.condition}{note}"
            )
        pending_text = "\n".join(pending_lines)
    else:
        pending_text = "  (all patients scheduled)"

    # --- Available slots (include doctor specialty for direct match) ---
    if obs.available_slots:
        slot_lines: List[str] = []
        shown = obs.available_slots[:25]
        for s in shown:
            specialty = doctor_specialties.get(s.doctor_id, "unknown")
            slot_lines.append(
                f"  {s.slot_id}: dr={s.doctor_id} ({specialty})"
                f"  {s.date}  {s.start_time}–{s.end_time}  [{s.time_window}]"
            )
        overflow = len(obs.available_slots) - len(shown)
        if overflow > 0:
            slot_lines.append(f"  … and {overflow} more available slots")
        slot_text = "\n".join(slot_lines)
    else:
        slot_text = "  (no slots available)"

    # --- Already confirmed bookings ---
    if obs.confirmed_bookings:
        booked_lines: List[str] = [
            f"  {b.patient_id} ({b.patient_name})"
            f" → {b.doctor_name} ({b.specialty})"
            f"  {b.date}  {b.start_time}–{b.end_time}"
            for b in obs.confirmed_bookings
        ]
        booked_text = "\n".join(booked_lines)
    else:
        booked_text = "  (none yet)"

    last_result_block = ""
    if obs.last_action_result:
        last_result_block = f"\nLast action result:\n  {obs.last_action_result}\n"

    return (
        f"=== Step {obs.step} ==={last_result_block}\n"
        f"PENDING PATIENTS ({len(obs.pending_requests)} remaining):\n{pending_text}\n\n"
        f"AVAILABLE SLOTS ({len(obs.available_slots)} total, showing up to 25):\n{slot_text}\n\n"
        f"CONFIRMED BOOKINGS ({len(obs.confirmed_bookings)}):\n{booked_text}\n\n"
        "Choose the best next action. Return ONLY a valid JSON object."
    )


# ---------------------------------------------------------------------------
# LLM call with exponential-backoff retries
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, prompt: str) -> Optional[str]:
    """
    Send the system + user prompt to the model and return the raw response
    text.  Retries up to MAX_RETRIES times with exponential backoff.
    Returns None if all attempts fail.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=256,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"    [llm] attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}")
            if attempt < MAX_RETRIES - 1:
                print(f"    [llm] retrying in {wait:.0f}s …")
                time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Action parsing — handles clean JSON and markdown-fenced responses
# ---------------------------------------------------------------------------

def parse_action(raw: str) -> Optional[SchedulerAction]:
    """
    Parse the model's raw text output into a validated SchedulerAction.

    Handles:
      - Bare JSON objects
      - Responses wrapped in ```json … ``` or ``` … ``` fences
    Returns None and prints a warning on any failure.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first line (``` or ```json) and last line (```)
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()

    try:
        data = json.loads(text)
        return SchedulerAction(**data)
    except Exception as exc:
        print(f"    [parse] failed — {exc}")
        print(f"    [parse] raw={raw[:250]!r}")
        return None


@dataclass
class _PatientProgress:
    viewed: bool = False
    emergency_flagged: bool = False
    specialty: Optional[str] = None
    availability_checked: bool = False


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM path cannot produce a valid action."""


def _normalize_text(*parts: Optional[str]) -> str:
    return " ".join(part.strip() for part in parts if part).lower()


def _needs_emergency_flag(request: PatientRequest) -> bool:
    text = _normalize_text(request.condition, request.notes)
    return any(hint in text for hint in _EMERGENCY_HINTS)


def _infer_specialty(request: PatientRequest) -> str:
    text = _normalize_text(request.condition, request.notes, request.required_specialty)

    for specialty, keywords in _SPECIALTY_RULES:
        if any(keyword in text for keyword in keywords):
            return specialty

    return request.required_specialty.lower().strip()


def _patient_priority_key(request: PatientRequest, progress: _PatientProgress) -> tuple:
    stage = 0
    if progress.viewed:
        stage = 1
    if progress.emergency_flagged:
        stage = 2
    if progress.specialty is not None:
        stage = 3
    if progress.availability_checked:
        stage = 4

    return (
        -stage,
        -_URGENCY_RANK.get(request.urgency, -1),
        request.preferred_date,
        request.preferred_time_window,
        request.patient_id,
    )


def _slot_sort_key(slot) -> tuple:
    return (slot.date, slot.start_time, slot.slot_id)


def _choose_slot_for_patient(
    request: PatientRequest,
    available_slots,
    specialty: str,
    doctor_specialties: Dict[str, str],
) -> Optional[object]:
    matching = [
        slot
        for slot in available_slots
        if doctor_specialties.get(slot.doctor_id, "").lower() == specialty.lower()
    ]
    if not matching:
        return None

    buckets = [
        [
            slot
            for slot in matching
            if slot.date == request.preferred_date
            and slot.time_window == request.preferred_time_window
        ],
        [slot for slot in matching if slot.date == request.preferred_date],
        [slot for slot in matching if slot.time_window == request.preferred_time_window],
        matching,
    ]

    for bucket in buckets:
        if bucket:
            return sorted(bucket, key=_slot_sort_key)[0]

    return sorted(matching, key=_slot_sort_key)[0]


class HeuristicAgent:
    """Deterministic offline policy used as the default baseline."""

    def __init__(self, state: Optional[object]) -> None:
        self._doctor_specialties: Dict[str, str] = {}
        self._progress: Dict[str, _PatientProgress] = {}
        if state is not None:
            self.reset(state)

    def reset(self, state: object) -> None:
        self._doctor_specialties = {
            doctor.doctor_id: doctor.specialty
            for doctor in getattr(state, "all_doctors", [])
        }
        self._progress = {}

    def next_action(self, obs: SchedulerObservation) -> SchedulerAction:
        pending = list(obs.pending_requests)
        if not pending:
            raise RuntimeError("heuristic agent called with no pending requests")

        for request in pending:
            self._progress.setdefault(request.patient_id, _PatientProgress())

        ordered = sorted(
            pending,
            key=lambda request: _patient_priority_key(
                request, self._progress[request.patient_id]
            ),
        )

        for request in ordered:
            progress = self._progress[request.patient_id]

            if not progress.viewed:
                progress.viewed = True
                return SchedulerAction(
                    action_type="view_request",
                    patient_id=request.patient_id,
                )

            if _needs_emergency_flag(request) and not progress.emergency_flagged:
                progress.emergency_flagged = True
                return SchedulerAction(
                    action_type="flag_emergency",
                    patient_id=request.patient_id,
                )

            if progress.specialty is None:
                progress.specialty = _infer_specialty(request)
                return SchedulerAction(
                    action_type="assign_specialist",
                    patient_id=request.patient_id,
                    specialty=progress.specialty,
                )

            if not progress.availability_checked:
                progress.availability_checked = True
                return SchedulerAction(action_type="check_availability")

            slot = _choose_slot_for_patient(
                request=request,
                available_slots=obs.available_slots,
                specialty=progress.specialty,
                doctor_specialties=self._doctor_specialties,
            )
            if slot is None:
                raise RuntimeError(
                    f"no available slot found for patient {request.patient_id}"
                )
            return SchedulerAction(
                action_type="schedule_slot",
                patient_id=request.patient_id,
                slot_id=slot.slot_id,
            )

        raise RuntimeError("heuristic agent could not determine a next action")


class LLMAgent:
    """LLM-backed policy preserved for optional remote runs."""

    def __init__(self, client: OpenAI, state: object) -> None:
        self._client = client
        self._doctor_specialties = {
            doctor.doctor_id: doctor.specialty
            for doctor in getattr(state, "all_doctors", [])
        }

    def next_action(self, obs: SchedulerObservation) -> SchedulerAction:
        prompt = build_observation_prompt(obs, self._doctor_specialties)
        raw = call_llm(self._client, prompt)
        if raw is None:
            raise LLMUnavailableError("LLM request failed after retries")

        action = parse_action(raw)
        if action is None:
            raise LLMUnavailableError("LLM response could not be parsed as an action")
        return action


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_task(
    mode: str,
    client: Optional[OpenAI],
    env: HealthcareAppointmentEnv,
    task_name: str,
    seed: int,
    global_start: float,
) -> float:
    """Run one full episode for `task_name` with `seed` under the selected mode."""

    divider = "─" * 62
    print(f"\n{'═' * 62}")
    print(f"  TASK : {task_name.upper()}   seed={seed}   mode={mode}")
    print(f"{'═' * 62}")

    obs = env.reset(task_name=task_name, seed=seed)
    print(f"  [reset] {obs.message}")

    init_state = env.state()
    n_patients = len(init_state.all_requests)

    heuristic_agent = HeuristicAgent(init_state)
    llm_agent = LLMAgent(client, init_state) if client is not None else None
    active_agent = heuristic_agent if mode == "heuristic" else llm_agent
    if active_agent is None:
        raise RuntimeError("LLM agent could not be initialised")

    step_count = 0
    final_reward = 0.0
    using_heuristic = mode == "heuristic"

    while not obs.done:
        if time.time() - global_start > GLOBAL_TIMEOUT_SECS:
            print("  [timeout] 18-minute wall-clock limit reached. Stopping task.")
            break

        if step_count >= MAX_EPISODE_STEPS:
            print(f"  [safeguard] Reached inference step cap ({MAX_EPISODE_STEPS}). Stopping.")
            break

        step_count += 1

        try:
            action = active_agent.next_action(obs)
        except LLMUnavailableError as exc:
            if mode == "auto" and not using_heuristic:
                print(f"  [auto] {exc}. Switching to heuristic mode.")
                heuristic_agent.reset(env.state())
                active_agent = heuristic_agent
                using_heuristic = True
                action = active_agent.next_action(obs)
            else:
                print(f"  [step {step_count:02d}] LLM unavailable: {exc}")
                break
        except Exception as exc:
            print(f"  [step {step_count:02d}] Agent error: {exc}")
            if mode == "auto" and not using_heuristic:
                print("  [auto] Falling back to heuristic mode after agent error.")
                heuristic_agent.reset(env.state())
                active_agent = heuristic_agent
                using_heuristic = True
                action = active_agent.next_action(obs)
            else:
                break

        try:
            result = env.step(action)
        except RuntimeError as exc:
            print(f"  [step {step_count:02d}] env.step() error: {exc}")
            break
        except Exception as exc:
            print(f"  [step {step_count:02d}] Unexpected env error: {exc}")
            break

        obs = result.observation
        final_reward = result.reward

        cov = result.info.get("coverage_so_far", "?")
        left = result.info.get("steps_remaining", "?")
        print(
            f"  [step {step_count:02d}] {action.action_type:<20s}"
            f"  reward={result.reward:.4f}"
            f"  done={str(result.done):<5s}"
            f"  cov={cov}  steps_left={left}"
        )
        if obs.last_action_result:
            snippet = obs.last_action_result[:110]
            ellipsis = "…" if len(obs.last_action_result) > 110 else ""
            print(f"             → {snippet}{ellipsis}")

    n_scheduled = n_patients - len(obs.pending_requests)
    print(divider)
    print(
        f"  steps_taken={step_count}"
        f"  scheduled={n_scheduled}/{n_patients}"
        f"  score={final_reward:.4f}"
    )
    return final_reward


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_mode = os.getenv("AGENT_MODE", DEFAULT_MODE).strip().lower()
    if default_mode not in SUPPORTED_MODES:
        default_mode = DEFAULT_MODE

    parser = argparse.ArgumentParser(description="Run the appointment scheduler agent.")
    parser.add_argument(
        "--mode",
        choices=sorted(SUPPORTED_MODES),
        default=default_mode,
        help="Agent mode: heuristic (default), llm, or auto.",
    )
    return parser.parse_args(argv)


def _build_llm_client() -> Optional[OpenAI]:
    missing = [
        v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN") if not os.environ.get(v)
    ]
    if missing:
        return None
    return OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    mode = args.mode

    print("═" * 62)
    print("  Healthcare Appointment Coordination — Inference Run")
    print("═" * 62)
    print(f"  MODE         : {mode}")
    print(f"  TASKS        : {' → '.join(TASK_ORDER)}")
    print(f"  SEEDS        : {TASK_SEEDS}")
    print(f"  TIMEOUT      : {GLOBAL_TIMEOUT_SECS}s wall-clock")
    print()

    client = None
    if mode in {"llm", "auto"}:
        client = _build_llm_client()
        if client is None:
            if mode == "llm":
                missing = [
                    v for v in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN")
                    if not os.environ.get(v)
                ]
                print(f"[FATAL] Missing required environment variable(s): {', '.join(missing)}")
                sys.exit(1)
            print("[INFO] LLM configuration missing; running heuristic mode instead.")
            mode = "heuristic"

    env = HealthcareAppointmentEnv()
    global_start = time.time()
    scores: Dict[str, float] = {}

    for task_name in TASK_ORDER:
        seed = TASK_SEEDS[task_name]
        try:
            score = run_task(mode, client, env, task_name, seed, global_start)
        except Exception as exc:
            print(f"[FATAL] Task '{task_name}' raised an unhandled exception: {exc}")
            score = 0.0
        scores[task_name] = score

    # --- Final report ---
    elapsed = time.time() - global_start
    avg     = sum(scores.values()) / max(len(scores), 1)

    print(f"\n{'═' * 62}")
    print("  FINAL RESULTS")
    print("═" * 62)
    for name, score in scores.items():
        filled  = int(score * 30)
        bar     = "█" * filled + "░" * (30 - filled)
        print(f"  {name:<8s}  {score:.4f}  |{bar}|")
    print(f"  {'─' * 46}")
    print(f"  {'average':<8s}  {avg:.4f}")
    print(f"  elapsed   {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print("═" * 62)

    # Non-zero exit if completely failed (likely misconfiguration)
    if avg < 0.001:
        print("\n[WARN] Average score is near zero — check API_BASE_URL / MODEL_NAME / HF_TOKEN.")
        sys.exit(2)


if __name__ == "__main__":
    main()
