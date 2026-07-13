#!/usr/bin/env python3
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

main = importlib.import_module("lexbrowser_eval.webarena_lite.cli").main


if __name__ == "__main__":
    raise SystemExit(main())
