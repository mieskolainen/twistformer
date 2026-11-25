#!/usr/bin/env bash
# Tau six-vector mixture training and checkpoint evaluation
#
# mikael.mieskolainen@cern.ch (2026)

set -eu
if [[ $# -ne 1 || -z "${1:-}" ]]
then
    printf 'Usage %s RUN_NAME\n' "${0##*/}" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
run_name="$1"
output="runs/$run_name"
python -m twistformer.cli.train --config configs/tau3_regression/transformer.yml --run-name="$run_name"
python -m twistformer.cli.evaluate --checkpoint "$output/checkpoints/best.pt" --split test
