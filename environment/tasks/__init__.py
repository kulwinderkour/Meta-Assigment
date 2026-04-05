from environment.tasks.task_easy import EasyTask
from environment.tasks.task_medium import MediumTask
from environment.tasks.task_hard import HardTask

TASK_REGISTRY: dict = {
    "easy":   EasyTask,
    "medium": MediumTask,
    "hard":   HardTask,
}

__all__ = ["EasyTask", "MediumTask", "HardTask", "TASK_REGISTRY"]
