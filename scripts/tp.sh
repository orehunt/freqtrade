#!/bin/sh

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/opts.sh

freqtrade test-pairlist -c $strategy \
     -c $live \
     -c $exchange \
     -c $amounts \
     -c $amounts_tuned \
     -c $pairlists \
     -c $paths \
     --quote $quote \
     --print-json \
     2>/dev/null | tail +2
