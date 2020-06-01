#!/bin/bash

[ -n "$BASH_SOURCE" ] && src=$BASH_SOURCE || src=$_
OPTS="$(realpath $(dirname $src))"
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
