from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parent


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


class Validator:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name=name, ok=ok, detail=detail))

    def run(self, name: str, fn: Callable[[], tuple[bool, str]]) -> None:
        try:
            ok, detail = fn()
            self.add(name, ok, detail)
        except Exception as exc:
            self.add(name, False, f"{type(exc).__name__}: {exc}")

    def has_failures(self) -> bool:
        return any(not r.ok for r in self.results)

    def print_report(self) -> None:
        print("=" * 72)
        print("Pre-submission validation")
        print("=" * 72)
        for r in self.results:
            status = "PASS" if r.ok else "FAIL"
            print(f"[{status}] {r.name}")
            if r.detail:
                print(f"       {r.detail}")
        print("-" * 72)
        passed = sum(1 for r in self.results if r.ok)
        total = len(self.results)
        print(f"Summary: {passed}/{total} checks passed")
        print("=" * 72)


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

def _file_exists(rel_path: str) -> tuple[bool, str]:
    path = ROOT / rel_path
    ok = path.exists() and path.is_file()
    return ok, str(path)


def check_openenv_yaml_exists() -> tuple[bool, str]:
    return _file_exists("openenv.yaml")


def check_inference_exists() -> tuple[bool, str]:
    return _file_exists("inference.py")


def check_task_files_exist() -> tuple[bool, str]:
    files = [
        "environment/tasks/task_easy.py",
        "environment/tasks/task_medium.py",
        "environment/tasks/task_hard.py",
    ]
    missing = [f for f in files if not (ROOT / f).is_file()]
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, "All 3 task files found"


def check_grader_files_exist() -> tuple[bool, str]:
    files = [
        "environment/graders/grader_easy.py",
        "environment/graders/grader_medium.py",
        "environment/graders/grader_hard.py",
    ]
    missing = [f for f in files if not (ROOT / f).is_file()]
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, "All 3 grader files found"


def check_dockerfile_exists() -> tuple[bool, str]:
    return _file_exists("Dockerfile")


def check_requirements_exists() -> tuple[bool, str]:
    return _file_exists("requirements.txt")


def check_env_example_docs_required_vars() -> tuple[bool, str]:
    env_file = ROOT / ".env.example"
    if not env_file.is_file():
        return False, ".env.example not found"

    required = {"API_BASE_URL", "MODEL_NAME", "HF_TOKEN"}
    declared: set[str] = set()

    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            declared.add(key)

    missing = sorted(required - declared)
    if missing:
        return False, f"Missing in .env.example: {', '.join(missing)}"
    return True, "Required env vars documented in .env.example"


# ---------------------------------------------------------------------------
# Runtime checks
# ---------------------------------------------------------------------------

def check_env_api_contract() -> tuple[bool, str]:
    from environment.env import HealthcareAppointmentEnv

    methods = ("reset", "step", "state")
    missing = [m for m in methods if not callable(getattr(HealthcareAppointmentEnv, m, None))]
    if missing:
        return False, f"Missing methods on HealthcareAppointmentEnv: {', '.join(missing)}"
    return True, "HealthcareAppointmentEnv exposes reset(), step(), state()"


def check_fastapi_routes() -> tuple[bool, str]:
    from environment.server import app

    required = {
        ("/health", "GET"),
        ("/reset", "POST"),
        ("/step", "POST"),
        ("/state", "GET"),
    }

    existing: set[tuple[str, str]] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if not path:
            continue
        for method in methods:
            existing.add((path, method))

    missing = sorted(required - existing)
    if missing:
        return False, f"Missing routes: {missing}"
    return True, "FastAPI app exposes /health, /reset, /step, /state"


def check_grader_outputs_float_range() -> tuple[bool, str]:
    from environment.env import HealthcareAppointmentEnv
    from environment.graders.grader_easy import EasyGrader
    from environment.graders.grader_medium import MediumGrader
    from environment.graders.grader_hard import HardGrader

    env = HealthcareAppointmentEnv()
    graders = {
        "easy": EasyGrader(),
        "medium": MediumGrader(),
        "hard": HardGrader(),
    }

    bad: list[str] = []
    for task_name, grader in graders.items():
        env.reset(task_name=task_name, seed=42)
        state = env.state()
        breakdown = grader.grade(state)

        fields = [
            "total",
            "coverage",
            "urgency_order",
            "specialty_match",
            "no_conflicts",
            "time_constraints",
        ]
        for field in fields:
            value = getattr(breakdown, field)
            if not isinstance(value, float):
                bad.append(f"{task_name}.{field} not float ({type(value).__name__})")
                continue
            if value < 0.0 or value > 1.0:
                bad.append(f"{task_name}.{field} out of range ({value})")

    if bad:
        return False, "; ".join(bad)
    return True, "All grader outputs are float values in [0.0, 1.0]"


def smoke_test_reset_and_one_step() -> tuple[bool, str]:
    from environment.env import HealthcareAppointmentEnv
    from environment.models import SchedulerAction

    env = HealthcareAppointmentEnv()
    obs = env.reset(task_name="easy", seed=42)

    if not obs.pending_requests:
        return False, "reset() returned no pending requests"
    if not obs.available_slots:
        return False, "reset() returned no available slots"

    patient = obs.pending_requests[0]

    # Pick the first slot with matching specialty if possible; fallback to first slot.
    state = env.state()
    doctor_specialty = {d.doctor_id: d.specialty.lower() for d in state.all_doctors}

    chosen_slot_id = None
    for sl in obs.available_slots:
        if doctor_specialty.get(sl.doctor_id) == patient.required_specialty.lower():
            chosen_slot_id = sl.slot_id
            break
    if chosen_slot_id is None:
        chosen_slot_id = obs.available_slots[0].slot_id

    action = SchedulerAction(
        action_type="schedule_slot",
        patient_id=patient.patient_id,
        slot_id=chosen_slot_id,
    )

    step_result = env.step(action)

    if step_result.observation.step < 1:
        return False, "step() did not advance observation.step"
    if not isinstance(step_result.reward, float):
        return False, "step() reward is not float"

    return True, f"reset()+step() smoke passed (reward={step_result.reward:.4f})"


def main() -> int:
    validator = Validator()

    validator.run("openenv.yaml exists", check_openenv_yaml_exists)
    validator.run("inference.py exists in root", check_inference_exists)
    validator.run("all 3 task files exist", check_task_files_exist)
    validator.run("all 3 grader files exist", check_grader_files_exist)
    validator.run("grader outputs are floats in [0.0, 1.0]", check_grader_outputs_float_range)
    validator.run("env exposes reset(), step(), state()", check_env_api_contract)
    validator.run("FastAPI exposes required endpoints", check_fastapi_routes)
    validator.run("Dockerfile exists", check_dockerfile_exists)
    validator.run("requirements.txt exists", check_requirements_exists)
    validator.run(
        "API_BASE_URL/MODEL_NAME/HF_TOKEN documented in .env.example",
        check_env_example_docs_required_vars,
    )
    validator.run("smoke test reset() and one step()", smoke_test_reset_and_one_step)

    validator.print_report()
    return 1 if validator.has_failures() else 0


if __name__ == "__main__":
    raise SystemExit(main())
