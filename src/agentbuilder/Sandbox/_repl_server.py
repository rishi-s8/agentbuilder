"""
Persistent REPL server that runs inside a Docker container.

Reads JSON commands from stdin (line-delimited), executes code in a
persistent globals() dict, and writes JSON results to stdout.
"""

import contextlib
import io
import json
import signal
import sys
import traceback


def timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def main():
    env = {}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            command = json.loads(line)
        except json.JSONDecodeError:
            result = {
                "stdout": "",
                "stderr": "Invalid JSON command",
                "success": False,
                "exit_code": 1,
            }
            sys.stdout.write(json.dumps(result) + "\n")
            sys.stdout.flush()
            continue

        code = command.get("code", "")
        timeout = command.get("timeout", 30)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            with contextlib.redirect_stdout(stdout_capture), \
                 contextlib.redirect_stderr(stderr_capture):
                exec(code, env)

            signal.alarm(0)

            result = {
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
                "success": True,
                "exit_code": 0,
            }
        except TimeoutError:
            signal.alarm(0)
            result = {
                "stdout": stdout_capture.getvalue(),
                "stderr": "Code execution timed out",
                "success": False,
                "exit_code": 124,
            }
        except Exception:
            signal.alarm(0)
            result = {
                "stdout": stdout_capture.getvalue(),
                "stderr": traceback.format_exc(),
                "success": False,
                "exit_code": 1,
            }

        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
