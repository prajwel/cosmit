#!/usr/bin/env python3
"""Command-line interface for Cosmit."""

import argparse
from pathlib import Path

from . import __version__, DEFAULT_THRESHOLD_FACTOR, CosmitError, process_observation


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cosmit",
        description=(
            "Identify cosmic ray affected event-list frames using a median + "
            "threshold_factor * NMAD threshold, remove flagged frames, "
            "and generate cleaned products and diagnostics."
        ),
    )
    parser.add_argument(
        "observation_dir",
        type=Path,
        help="Root directory of the UVIT observation to process.",
    )
    parser.add_argument(
        "--input-source",
        choices=("v7", "resolvit"),
        default="v7",
        help="Input product source (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold-factor",
        type=float,
        default=DEFAULT_THRESHOLD_FACTOR,
        help=(
            "Multiplier applied to the normal-scaled median absolute deviation "
            "in the frame-count threshold (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv=None):
    """Run Cosmit from the command line."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.observation_dir.is_dir():
        parser.error(
            "observation directory does not exist or is not a directory: "
            f"{args.observation_dir}"
        )

    try:
        results = process_observation(
            args.observation_dir,
            input_source=args.input_source,
            threshold_factor=args.threshold_factor,
        )
    except (CosmitError, FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: error: {exc}\n")

    print(f"\nCosmit {__version__}: processing complete")
    print(f"Input source: {args.input_source}")
    print(f"Event lists processed: {len(results)}")

    for paths, summary in results:
        print(f"\n{paths.product_id}")
        print(f"  Cleaned event list : {paths.cleaned_events_list}")
        print(
            f"  Frames flagged     : {summary['Flagged frames']} "
            f"({summary['Flagged frames fraction']:.4f}%)"
        )
        print(
            f"  Events removed     : {summary['Removed events']} "
            f"({summary['Removed events fraction']:.4f}%)"
        )
        print(f"  Exposure scale     : {summary['Exposure scale factor']:.6f}")
        print(f"  Diagnostics        : {paths.diagnostics_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
