from typing import Any
from .executor_types import ExecuteResult, Executor



class Game24Executor(Executor):
    def execute(self, func: str, tests: list, timeout: int = 5) -> ExecuteResult:
        """
        Stubbed out—Game24 doesn’t use `execute` in the same way.
        """
        # We don’t bundle multiple tests here, so we just mark it unimplemented.
        raise NotImplementedError("Game24Executor.execute is not used for evaluation.")

    def evaluate(self, expr: str, timeout: int = 5) -> bool:
        """
        Evaluate a 24-game expression and return True if it equals 24.

        Args:
            name (str): Identifier for the puzzle (unused).
            expr (str): A Python expression using +, -, *, / on the four numbers.
                        e.g. "11+11+1+1"
            timeout (int): Not used here (no infinite loops expected).

        Returns:
            bool: True if eval(expr) == 24 (within a small tolerance), else False.
        """
        try:
            # Compute the numeric result
            result: Any = eval(expr, {"__builtins__": {}}, {})
            # Allow for floating‐point rounding
            return abs(float(result) - 24.0) < 1e-8
        except Exception:
            return False
