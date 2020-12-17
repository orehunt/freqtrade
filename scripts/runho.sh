#!/usr/bin/env bash
set -euo pipefail

[ -n "$INST" ] && inst="-inst $INST" || inst=
[ -n "$RISK" ] && risk="-risk $RISK" || risk=
[ -n "$STACK" ] && stack="-stack" || stack=

ho.py -i $TIMEFRAME \
    -b \
    -lo $LOSSF \
    -alg $ALGO \
    -sgn $SPACES \
    $risk \
    -amt on:roi,trailing,stoploss \
    $stack \
    -mt ${MT:-1} \
    -mx ${MX:-1} \
    -pts ${POINTS:-1} \
    -rpt ${RPT:-10} \
    -j $JOBS -res \
    -e $EPOCHS \
    -mode $MODE \
    $TDATE \
    $inst $@
