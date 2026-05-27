"""Run validation-based SETTA configuration selection before main experiments."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from config import DEFAULT_VALIDATION_SELECTED_AB_PATH, SETTA_FIXED_PARAMS, get_setta_config


DEFAULT_MAIN_SCRIPTS = [
    "exp1_main_results.py",
    "exp2_scalability_robustness.py",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SETTA validation selection, then run main experiment scripts."
    )
    parser.add_argument(
        "--selected-ab-path",
        default=DEFAULT_VALIDATION_SELECTED_AB_PATH,
        help="Path to validation-selected alpha/beta summary CSV.",
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="Skip validation search and reuse an existing selected alpha/beta CSV.",
    )
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=DEFAULT_MAIN_SCRIPTS,
        help="Experiment scripts to run after selection.",
    )
    parser.add_argument(
        "--selection-script",
        default="exp_val_ab_risk_selection_constrained_v2.py",
        help="Validation-selection script to run first.",
    )
    return parser.parse_args()


def run_command(cmd, env):
    print("\n" + "=" * 88)
    print("Running:", " ".join(cmd))
    print("=" * 88)
    subprocess.run(cmd, check=True, env=env)


def print_selected_configs(selected_ab_path):
    print("\nSelected SETTA configurations")
    print(f"alpha/beta CSV: {selected_ab_path}")
    for dataset in SETTA_FIXED_PARAMS:
        cfg = get_setta_config(dataset, selected_ab_path)
        print(
            f"  {dataset}: alpha={cfg['alpha']}, beta={cfg['beta']}, "
            f"k={cfg['k']}, metric={cfg['metric']}, num_props={cfg['num_props']}"
        )


def main():
    args = parse_args()
    selected_ab_path = Path(args.selected_ab_path)
    env = os.environ.copy()
    env["SETTA_SELECTED_AB_PATH"] = str(selected_ab_path)

    if not args.skip_selection:
        output_dir = selected_ab_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                sys.executable,
                args.selection_script,
                "--output-dir",
                str(output_dir),
            ],
            env=env,
        )
    elif not selected_ab_path.exists():
        raise FileNotFoundError(
            "Validation-selected alpha/beta file not found. "
            "Run without --skip-selection or provide --selected-ab-path. "
            f"Expected path: {selected_ab_path}"
        )

    print_selected_configs(str(selected_ab_path))

    for script in args.scripts:
        run_command([sys.executable, script], env=env)


if __name__ == "__main__":
    main()
