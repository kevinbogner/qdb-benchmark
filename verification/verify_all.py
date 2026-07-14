#!/usr/bin/env python3
"""Run the exact MF certificate verifiers for every supported DV protocol."""

from __future__ import annotations

import argparse

from . import e91_exact_witness, qdb2019_exact_witness, rational_exact_witnesses

PROTOCOLS = (
    "qdb2017",
    "qdb2019",
    "mutual-entanglement",
    "e91",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        choices=(*PROTOCOLS, "all"),
        default="all",
    )
    args = parser.parse_args(argv)

    selected = PROTOCOLS if args.protocol == "all" else (args.protocol,)
    for index, protocol in enumerate(selected):
        if index:
            print()
        if protocol == "qdb2017":
            rational_exact_witnesses.verify_qdb2017()
        elif protocol == "qdb2019":
            qdb2019_exact_witness.main()
        elif protocol == "mutual-entanglement":
            rational_exact_witnesses.verify_mutual()
        elif protocol == "e91":
            e91_exact_witness.main()
        else:  # pragma: no cover - argparse enforces the choices
            raise AssertionError(protocol)


if __name__ == "__main__":
    main()
