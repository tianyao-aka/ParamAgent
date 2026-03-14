import ast
import signal
import astunparse
import traceback
import sys
import subprocess
import tempfile
import os
from .executor_utils import function_with_timeout

from typing import List
from .executor_types import ExecuteResult, Executor

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            try:
                # Use isolated namespace to prevent memory leak from accumulating globals
                local_ns = {}
                function_with_timeout(exec, (func_test_list[i], {"__builtins__": __builtins__}, local_ns), timeout)
                success_tests += [tests[i]]
                
            except Exception:
                output = get_output(func, tests[i], timeout=timeout)
                failed_tests += [f"{tests[i]} # output: {output}"]
                is_passing = False

        state = []
        for test in tests:
            if test in success_tests:
                state += [True]
            else:
                state += [False]

        state = tuple(state)

        feedback = "Tested passed:"
        for test in success_tests:
            feedback += f"\n{test}"
        feedback += "\n\nTests failed:"
        for test in failed_tests:
            feedback += f"\n{test}"

        return ExecuteResult(is_passing, feedback, state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        code = f"""{func}

{test}

check({name})
    """
        try:
            print ('-------------------eval code-------------------------------')
            print (code)
            # Use isolated namespace to prevent memory leak from accumulating globals
            local_ns = {}
            function_with_timeout(exec, (code, {"__builtins__": __builtins__}, local_ns), timeout)
            return True
        except Exception as e:
            # err = traceback.format_exc()
            # print (err)
            return False

    def evaluate_livecodebench(self, name: str, program: str, test_cases: List, timeout: int = 10) -> bool:
        """
        Evaluates a complete Python program against stdin/stdout test cases.

        Args:
            name: Problem identifier (for logging)
            program: Complete Python program code
            test_cases: List of TestCase(input, output, testtype) from livecodebench_utils
            timeout: Execution timeout per test (seconds)

        Returns:
            True if all test cases pass, False otherwise
        """
        temp_file = None
        try:
            for i, test in enumerate(test_cases):
                try:
                    # Write program to temporary file
                    with tempfile.NamedTemporaryFile(
                        mode='w',
                        suffix='.py',
                        delete=False,
                        encoding='utf-8'
                    ) as f:
                        f.write(program)
                        temp_file = f.name

                    # Run with stdin input
                    result = subprocess.run(
                        ['python', temp_file],
                        input=test.input,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )

                    # Check for runtime errors
                    if result.returncode != 0:
                        print(f"Test {i} failed with return code {result.returncode}")
                        if result.stderr:
                            print(f"  stderr: {result.stderr[:200]}")
                        return False

                    # Compare output (exact match after strip, like HumanEval assertions)
                    actual = result.stdout.strip()
                    expected = test.output.strip()

                    if actual != expected:
                        print(f"Test {i} failed for problem '{name}':")
                        print(f"  Input: {test.input[:100].strip()}{'...' if len(test.input) > 100 else ''}")
                        print(f"  Expected: {expected[:100]}{'...' if len(expected) > 100 else ''}")
                        print(f"  Actual: {actual[:100]}{'...' if len(actual) > 100 else ''}")
                        return False

                    # Clean up temp file after each test
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
                        temp_file = None

                except subprocess.TimeoutExpired:
                    print(f"Test {i} timed out after {timeout}s for problem '{name}'")
                    return False
                except Exception as e:
                    print(f"Test {i} error for problem '{name}': {e}")
                    return False
                finally:
                    # Clean up temp file
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
                        temp_file = None

            # All tests passed
            print(f"All {len(test_cases)} tests passed for problem '{name}'")
            return True

        except Exception as e:
            print(f"Unexpected error evaluating problem '{name}': {e}")
            return False
        finally:
            # Final cleanup
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)

def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        # Use isolated namespace to prevent memory leak from accumulating globals
        local_ns = {}
        exec(f"from typing import *\n{func}", {"__builtins__": __builtins__}, local_ns)
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, {"__builtins__": __builtins__}, local_ns), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pass
    # Test the function
    func = "def add(a, b):\n    while True:\n        x = 1\n    return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(1, 2) == 4"]
    print(PyExecutor().execute(func, tests, timeout=1))


