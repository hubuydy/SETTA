"""Compatibility entrypoint for validation-based SETTA alpha/beta selection."""

import runpy


if __name__ == "__main__":
    runpy.run_path("exp_validation-based_selection.py", run_name="__main__")
