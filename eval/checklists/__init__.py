"""
eval/checklists/ — Per-dimension checklist definitions and two-tier pass/fail logic.

Each submodule exports:
  CHECKLIST_TEMPLATES : list[ChecklistItem]   — templates with result=False placeholder
  evaluate_<dimension> : (items) -> DimensionResult

Imports for convenience:
"""

from eval.checklists.faithfulness import (
    CHECKLIST_TEMPLATES as FAITHFULNESS_TEMPLATES,
    evaluate_faithfulness,
)
from eval.checklists.answer_relevancy import (
    CHECKLIST_TEMPLATES as ANSWER_RELEVANCY_TEMPLATES,
    evaluate_answer_relevancy,
)
from eval.checklists.context_precision import (
    CHECKLIST_TEMPLATES as CONTEXT_PRECISION_TEMPLATES,
    evaluate_context_precision,
)
from eval.checklists.context_recall import (
    CHECKLIST_TEMPLATES as CONTEXT_RECALL_TEMPLATES,
    evaluate_context_recall,
)

__all__ = [
    "FAITHFULNESS_TEMPLATES",
    "evaluate_faithfulness",
    "ANSWER_RELEVANCY_TEMPLATES",
    "evaluate_answer_relevancy",
    "CONTEXT_PRECISION_TEMPLATES",
    "evaluate_context_precision",
    "CONTEXT_RECALL_TEMPLATES",
    "evaluate_context_recall",
]
