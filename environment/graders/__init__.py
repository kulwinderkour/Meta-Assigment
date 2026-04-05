from environment.graders.grader_easy import EasyGrader
from environment.graders.grader_medium import MediumGrader
from environment.graders.grader_hard import HardGrader

GRADER_REGISTRY: dict = {
    "easy":   EasyGrader,
    "medium": MediumGrader,
    "hard":   HardGrader,
}

__all__ = ["EasyGrader", "MediumGrader", "HardGrader", "GRADER_REGISTRY"]
