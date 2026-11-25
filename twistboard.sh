#!/bin/sh
# Start Twistboard for the repository runs
#
# mikael.mieskolainen@cern.ch (2026)

cd "$(dirname -- "$0")" || exit 1
python -m twistboard serve --root runs --open --force &

