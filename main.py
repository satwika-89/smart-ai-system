#!/usr/bin/env python3
"""Top-level CLI runner for common project tasks.

Usage examples:
  python main.py streamlit
  python main.py load
  python main.py preprocess
  python main.py train --target "C6H6(GT)"

This is a small convenience wrapper that calls the existing module entrypoints.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]

def run_streamlit():
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py")]
    subprocess.run(cmd)

def run_loader():
    cmd = [sys.executable, "-m", "src.data.aqi_loader"]
    subprocess.run(cmd, cwd=str(ROOT))

def run_preprocess():
    cmd = [sys.executable, "-m", "src.preprocessing.airquality_pipeline"]
    subprocess.run(cmd, cwd=str(ROOT))

def run_train(target: str, advanced: bool = True):
    module = "src.models.train_pollution_advanced" if advanced else "src.models.train_pollution"
    cmd = [sys.executable, "-m", module, "--target", target]
    subprocess.run(cmd, cwd=str(ROOT))

def main():
    p = argparse.ArgumentParser(prog="main.py", description="Project task runner")
    p.add_argument("cmd", choices=["streamlit","load","preprocess","train"], help="command to run")
    p.add_argument("--target", default="C6H6(GT)", help="target column for training")
    p.add_argument("--simple", action="store_true", help="use simple train instead of advanced")
    args = p.parse_args()

    if args.cmd == "streamlit":
        run_streamlit()
    elif args.cmd == "load":
        run_loader()
    elif args.cmd == "preprocess":
        run_preprocess()
    elif args.cmd == "train":
        run_train(args.target, advanced=not args.simple)

if __name__ == "__main__":
    main()
