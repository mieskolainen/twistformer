#!/usr/bin/env bash
# Top tagging with sheaf and Gram observables
#
# mikael.mieskolainen@cern.ch (2026)

set -eu
run_name="${1-toptag}"
if [[ $# -gt 1 || -z "$run_name" ]]
then
    printf 'Usage %s [RUN_NAME]\n' "${0##*/}" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
output="runs/$run_name"
python -m twistformer.cli.train \
    --config configs/toptag/twistgraph_sheaf_gram.yml --output "$output" --device cuda
python -m twistformer.cli.evaluate \
    --checkpoint "$output/checkpoints/best.pt" --split test --device cuda
