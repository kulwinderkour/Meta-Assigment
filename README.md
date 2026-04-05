# healthcare-appointment-coordination-env

Submission-oriented OpenEnv environment where a deterministic heuristic baseline coordinates clinic appointments under specialty, urgency, and scheduling constraints, with optional LLM and auto-fallback modes for remote runs.

## Environment Description and Motivation

This project simulates a realistic outpatient scheduling workflow:
- Patients have symptoms, urgency, preferences, and required specialties.
- Doctors have specialties and constrained availability.
- The agent must make sequential decisions via a strict action API.

Motivation: benchmark whether an agent can do **structured operational reasoning** (triage, specialty routing, urgency ordering, emergency escalation, and conflict avoidance) rather than generic chat.

## Action Space

The agent outputs one `SchedulerAction` JSON per step.

Allowed actions:

```json
{"action_type": "view_request",      "patient_id": "p001"}
{"action_type": "check_availability", "doctor_id": "d001"}
{"action_type": "assign_specialist",  "patient_id": "p001", "specialty": "neurology"}
{"action_type": "schedule_slot",      "patient_id": "p001", "slot_id": "s001"}
{"action_type": "reschedule_slot",    "patient_id": "p001", "slot_id": "s001", "new_slot_id": "s002"}
{"action_type": "flag_emergency",     "patient_id": "p004"}
{"action_type": "confirm_booking",    "patient_id": "p001", "slot_id": "s001"}
```

Validation is enforced by Pydantic + environment checks.

## Observation Space

`SchedulerObservation` (agent-visible view) includes:
- `pending_requests`: unscheduled patient requests
- `available_slots`: currently unbooked slots
- `confirmed_bookings`: bookings already confirmed
- `message`: environment guidance/status
- `step`, `done`, `last_action_result`

The full internal state is only exposed through `state()` for grading/debugging.

## Reward Design

Reward combines:
- **Shaped step reward** for progress (e.g., valid scheduling decisions)
- **Penalties** for invalid/wasteful actions
- **Terminal score** from task grader (`RewardBreakdown`, range `[0.0, 1.0]`)

Core rubric axes used by graders:
- coverage
- specialty_match
- urgency_order
- no_conflicts
- time_constraints

## Task Descriptions

| Task | Patients | Doctors | Slots | Main challenge |
|---|---:|---:|---:|---|
| easy | 1 | 3 | 6 | Correct specialty + preferred time window |
| medium | 4 | 4 | 12 | Scarce cardiology slots + urgency ordering |
| hard | 8 | 5 | 20 | Emergency escalation + rescheduling + routing traps |

Highlights:
- `easy`: deterministic single correct path (plus acceptable fallback).
- `medium`: critical cardiology patient must get earliest cardiology slot.
- `hard`: `p004` must be emergency-flagged; scarce specialist slots force ordering and possible rescheduling.

## Grader Methodology

Each task has a deterministic programmatic grader:
- `environment/graders/grader_easy.py`
- `environment/graders/grader_medium.py`
- `environment/graders/grader_hard.py`

Graders consume `SchedulerState` and output `RewardBreakdown(total, ...)` in `[0.0, 1.0]`.

No LLM grading is used.

## Local Setup

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Configure environment variables:

```bash
cp .env.example .env
```

Set in `.env` only if you plan to use `llm` or `auto` mode:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

## How to Run `inference.py`

```bash
python inference.py
```

This runs all tasks in deterministic order: `easy -> medium -> hard` with fixed seeds using the default deterministic heuristic baseline.

To use the remote model path instead:

```bash
python inference.py --mode llm
python inference.py --mode auto
```

## Baseline Wrapper

`baseline/run_baseline.py` is intentionally a thin wrapper that calls the same `main()` logic in `inference.py`, which defaults to the reproducible heuristic baseline.

```bash
python baseline/run_baseline.py
```

## How to Validate `openenv.yaml`

Use the OpenEnv CLI validator:

```bash
openenv validate openenv.yaml
```

## Docker Build and Run

```bash
docker build -t healthcare-appointment-coordination-env .
docker run --rm -p 7860:7860 --env-file .env healthcare-appointment-coordination-env
```

Server endpoints:
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

## HF Spaces Deployment Instructions

1. Create a new **Docker Space** on Hugging Face.
2. Push this repository (including `Dockerfile`, `openenv.yaml`, `environment/server.py`).
3. In Space settings, add secrets:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. Ensure exposed port is `7860` (default in Dockerfile).
5. Verify health check:
   - `GET /health` returns HTTP 200.

## Expected Baseline Output Format

Typical run output:

```text
==============================================================
  Healthcare Appointment Coordination — Inference Run
==============================================================
  API_BASE_URL : https://api-inference.huggingface.co/v1
  MODEL_NAME   : meta-llama/Llama-3.3-70B-Instruct
  TASKS        : easy -> medium -> hard

==============================================================
  TASK : EASY   seed=42
==============================================================
  [step 01] schedule_slot ... reward=...
  ...

==============================================================
  FINAL RESULTS
==============================================================
  easy      0.xxxx  |██████....|
  medium    0.xxxx  |█████.....|
  hard      0.xxxx  |████......|
  ----------------------------------------------
  average   0.xxxx
```

## API Summary

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | HF health check |
| POST | `/reset` | Start task episode |
| POST | `/step` | Apply one agent action |
| GET | `/state` | Full internal state |
