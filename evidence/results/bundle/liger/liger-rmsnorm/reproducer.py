import sys
sys.dont_write_bytecode = True

import json
from pathlib import Path

from daggertrace.targets import run_liger_reproducer

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "external" / "liger" / "tree"
    result = run_liger_reproducer(target)
    print(json.dumps(result, sort_keys=True))
