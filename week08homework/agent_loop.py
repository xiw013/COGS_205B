#!/usr/bin/env python3
"""
Programmatic prompt -> generate -> test -> feedback loop using GeminiSimpleAPI.

Companion script for 071-programmatic-access.md.

Setup:
    pip install numpy
    export GEMINI_API_KEY=...

Usage:
    python iterating_regression.py
"""

import subprocess
import sys
import shutil
from pathlib import Path

# Allow running from this directory without installing the package.
_FILES_DIR = Path(__file__).resolve().parent.parent
if str(_FILES_DIR) not in sys.path:
    sys.path.insert(0, str(_FILES_DIR))

from gemini_simple_api import GeminiSimpleAPI  # noqa: E402

TASK_DIR = Path(__file__).parent
TEST_DIR = TASK_DIR / "tests"
TEST_FILE = TEST_DIR / "test_bayes_factor.py"
SOURCE_FILE = TASK_DIR / "bayes_factor.py"
# Set test_**n.py to read-only(!)
# Note this won't do anything if the agent can run as root.
TEST_FILE.chmod(0o444)

# Modifiable parameters

MAX_ATTEMPTS = 3
INCLUDE_TEST_FILE = True


PROMPT_FILE = TASK_DIR / "task.txt"

def run_tests() -> tuple[int, str]:
    result = subprocess.run(
        ["python3", "-m", "unittest", "discover", "-s", TEST_DIR],
        cwd=TASK_DIR,
        capture_output=True,
        text=True,
    )
    return result.returncode, (result.stdout + result.stderr).strip()

print(TASK_DIR)
client = GeminiSimpleAPI(
    api_key_file="../../secrets/gemini.json",
    model="gemma-4-31b-it",
    working_dir=TASK_DIR,
    # protected_directories=[TEST_DIR],
)

prompt_text = PROMPT_FILE.read_text()

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"\n=== Attempt {attempt} ===")
    files, notes = client.prompt(
        prompt=prompt_text,
        attachments=[SOURCE_FILE, TEST_FILE] if INCLUDE_TEST_FILE else [],
        verbose=True,
    )

    # Here you could re-insert the test file if it was modified.

    code, output = run_tests()
    print(f"Output: {output}")

    # Archive the attempt
    (TASK_DIR / f"attempt_{attempt}").mkdir(parents=True, exist_ok=True)
    (TASK_DIR / f"attempt_{attempt}" / "agent_loop_output.txt").write_text(output)
    (TASK_DIR / f"attempt_{attempt}" / "task.txt").write_text(prompt_text)
    for file in files:
        shutil.copy(file, TASK_DIR / f"attempt_{attempt}" / file.name)

    # input("Press Enter to continue...")
    if code == 0:
        print(f"\nTests passed on attempt {attempt}.")
        break
    prompt_text += (
        f"\n\n## Attempt {attempt} failed\n"
        f"```\n{output}\n```\n"
        "Fix the failures above."
    )
else:
    print(f"\nStopped after {MAX_ATTEMPTS} attempts; tests still failing.")
    sys.exit(1)