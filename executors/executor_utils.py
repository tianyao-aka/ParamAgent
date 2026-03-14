import multiprocessing
from multiprocessing import Process, Queue


def timeout_handler(_, __):
    raise TimeoutError()

import os, json
def to_jsonl(dict_data, file_path):
    with open(file_path, 'a') as file:
        json_line = json.dumps(dict_data)
        file.write(json_line + os.linesep)

from threading import Thread
class PropagatingThread(Thread):
    def run(self):
        self.exc = None
        try:
            if hasattr(self, '_Thread__target'):
                # Thread uses name mangling prior to Python 3.
                self.ret = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)
            else:
                self.ret = self._target(*self._args, **self._kwargs)
        except BaseException as e:
            self.exc = e

    def join(self, timeout=None):
        super(PropagatingThread, self).join(timeout)
        if self.exc:
            raise self.exc
        return self.ret
    

def function_with_timeout(func, args, timeout):
    """
    Run func(*args) in a separate thread and enforce a timeout.
    Thread-based implementation that works on all platforms (macOS, Linux, Windows).

    Note: This won't kill CPU-bound infinite loops, but will correctly timeout
    and raise TimeoutError for I/O-bound operations and most Python code.
    """
    result_container = []

    def wrapper():
        result_container.append(func(*args))

    thread = PropagatingThread(target=wrapper)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"Function {func.__name__!r} timed out after {timeout} s")
    else:
        return result_container[0]


# COMMENTED OUT: Multiprocessing implementation that doesn't work on macOS
# because nested _worker function cannot be pickled with 'spawn' start method
#
# def function_with_timeout(func, args, timeout):
#     """
#     Run func(*args) in a separate process and enforce a wall-clock timeout.
#
#     Inputs:
#       • func    – a picklable callable to invoke.
#       • args    – a tuple or list of positional arguments for func.
#       • timeout – maximum number of seconds to wait.
#
#     Returns:
#       The return value of func(*args), if it completes within `timeout` seconds.
#
#     Raises:
#       TimeoutError:   if func does not finish within timeout.
#       Exception:      re-raises any exception that func itself raised.
#     """
#     # A queue to get back either (True, result) or (False, exception)
#     result_queue = Queue()
#
#     def _worker(q, fn, fn_args):
#         try:
#             res = fn(*fn_args)
#             q.put((True, res))
#         except Exception as e:
#             q.put((False, e))
#
#     # Spawn the worker process
#     proc = Process(target=_worker, args=(result_queue, func, tuple(args)))
#     proc.daemon = True
#     proc.start()
#
#     # Wait up to `timeout` seconds
#     proc.join(timeout)
#
#     if proc.is_alive():
#         # Still running → timeout: kill it
#         proc.terminate()
#         proc.join()
#         raise TimeoutError(f"Function {func.__name__!r} timed out after {timeout} s")
#
#     # Otherwise, collect the result
#     success, payload = result_queue.get()
#     if success:
#         return payload
#     else:
#         # Worker caught and sent us an exception: re-raise it here
#         raise payload


# Py tests

# if __name__ == "__main__":
#     formatter = PySubmissionFormatter()
#     leetcode_1 = 'class Solution:\n    def solveSudoku(self, board: List[List[str]]) -> None:\n        """\n        Do not return anything, modify board in-place instead.\n        """\n        '
#     humaneval_1 = 'def solveSudoku(self, board: List[List[str]]) -> None:\n        """\n        Do not return anything, modify board in-place instead.\n        """\n'

#     assert leetcode_1 == formatter.to_leetcode(humaneval_1)
#     assert humaneval_1 == formatter.to_humaneval(leetcode_1)




