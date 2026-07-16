"""Continual Learning with EWC (Elastic Weight Consolidation)."""
import numpy as np
from typing import Optional
from loguru import logger


class ContinualLearner:
    """Prevent catastrophic forgetting with EWC."""

    def __init__(self, ewc_lambda: float = 1000.0):
        self.ewc_lambda = ewc_lambda
        self._fisher_information: dict[str, np.ndarray] = {}
        self._optimal_params: dict[str, np.ndarray] = {}
        self._task_count = 0

    def consolidate(self, task_name: str, params: dict[str, np.ndarray]):
        """Save parameters after learning a task."""
        if self._task_count > 0:
            # Compute Fisher information (simplified)
            for name, param in params.items():
                if name in self._optimal_params:
                    fisher = (param - self._optimal_params[name]) ** 2
                    if name in self._fisher_information:
                        self._fisher_information[name] = np.maximum(
                            self._fisher_information[name], fisher
                        )
                    else:
                        self._fisher_information[name] = fisher
                self._optimal_params[name] = param.copy()
        else:
            for name, param in params.items():
                self._optimal_params[name] = param.copy()
                self._fisher_information[name] = np.zeros_like(param)
        self._task_count += 1
        logger.info(f"EWC consolidated task {task_name} (total: {self._task_count})")

    def penalty(self, current_params: dict[str, np.ndarray]) -> float:
        """Compute EWC penalty for current parameters."""
        loss = 0.0
        for name, param in current_params.items():
            if name in self._fisher_information and name in self._optimal_params:
                loss += float(np.sum(
                    self._fisher_information[name] * (param - self._optimal_params[name]) ** 2
                ))
        return self.ewc_lambda * loss

    def stats(self) -> dict:
        return {"tasks_consolidated": self._task_count, "lambda": self.ewc_lambda}
