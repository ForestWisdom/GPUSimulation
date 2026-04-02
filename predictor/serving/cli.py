"""CLI entrypoint for demo inference."""

from __future__ import annotations

import argparse

from predictor.serving.predict import KernelLatencyPredictor


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for demo kernel inference."""

    parser = argparse.ArgumentParser(description="Predict a single kernel latency.")
    parser.add_argument("--name", default="demo_kernel")
    parser.add_argument("--family", dest="family_hint", default=None)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--backend", default="cuda")
    parser.add_argument(
        "--dimension",
        action="append",
        default=[],
        help="Dimension in key=value form. Can be repeated.",
    )
    return parser


def main() -> None:
    """Run the placeholder latency predictor from the command line."""

    parser = build_parser()
    args = parser.parse_args()
    dimensions = _parse_dimensions(args.dimension)
    predictor = KernelLatencyPredictor.default()
    prediction = predictor.predict_from_raw_metadata(
        {
            "name": args.name,
            "family_hint": args.family_hint,
            "dtype": args.dtype,
            "backend": args.backend,
            "dimensions": dimensions,
        }
    )
    print(prediction)


def _parse_dimensions(raw_dimensions: list[str]) -> dict[str, int]:
    """Parse repeated CLI dimension arguments into a dictionary."""

    dimensions: dict[str, int] = {}
    for item in raw_dimensions:
        key, _, raw_value = item.partition("=")
        if key and raw_value:
            dimensions[key] = int(raw_value)
    return dimensions
