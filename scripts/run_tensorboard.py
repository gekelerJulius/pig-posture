"""Run TensorBoard for compact PigPose training logs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pigpose.config import OUTPUTS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help=(
            "TensorBoard log directory. Defaults to outputs/ so TensorBoard can "
            "discover newly created runs while it is running."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument(
        "--reload-interval",
        type=int,
        default=5,
        help="Seconds between TensorBoard reload scans for new event files.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available TensorBoard log roots and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        list_log_roots()
        return

    logdir = args.logdir or OUTPUTS_DIR
    if args.logdir is None:
        logdir.mkdir(parents=True, exist_ok=True)
    elif not logdir.exists():
        raise FileNotFoundError(
            "No TensorBoard log directory found. Run training first or pass --logdir."
        )

    print(f"TensorBoard logdir: {logdir}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(logdir),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--reload_interval",
            str(args.reload_interval),
        ],
        check=True,
    )


def list_log_roots() -> None:
    roots = tensorboard_event_files()
    if not roots:
        print(f"No TensorBoard event files found below {OUTPUTS_DIR}")
        return
    logdirs = sorted({event.parent for event in roots})
    for logdir in logdirs:
        print(logdir)


def tensorboard_event_files() -> list[Path]:
    return sorted(OUTPUTS_DIR.glob("**/tensorboard/**/events.out.tfevents*"))


if __name__ == "__main__":
    main()
