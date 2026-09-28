"""
run_all.py - Master execution script for FlyRank content decay research pipeline.
"""

import os
import sys
import subprocess


def run_cmd(cmd, description):
    print(f"\n{'='*70}\n[RUNNING] {description}\nCommand: {cmd}\n{'='*70}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[FAILED] {description} exited with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[PASSED] {description}")


def main():
    py_bin = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))

    # Auto-detect local virtual environment if invoked with global python
    venv_win = os.path.join(root_dir, ".venv", "Scripts", "python.exe")
    venv_nix = os.path.join(root_dir, ".venv", "bin", "python")
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        if os.path.exists(venv_win):
            py_bin = venv_win
        elif os.path.exists(venv_nix):
            py_bin = venv_nix

    # Step 1: Feature preparation & Data Contract verification
    run_cmd(f'"{py_bin}" "{os.path.join(script_dir, "01_prepare_features.py")}"', "Feature Engineering & Data Contract Assertions")

    # Step 2: Model Training, Validation, and Artifact Exports
    run_cmd(f'"{py_bin}" "{os.path.join(script_dir, "02_train_models.py")}"', "5-Fold Client-Grouped CV, Scorecard, Queues & Charts")

    print("\nAll production pipeline stages finished successfully.")


if __name__ == "__main__":
    main()
