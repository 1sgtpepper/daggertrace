import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from daggertrace.art_target import run_art_reproducer

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "external" / "openpipe-art" / "tree"
    print(json.dumps(run_art_reproducer(target), sort_keys=True))
