#!/usr/bin/env bash
set -euo pipefail

[ -n "$INST" ] && inst="-inst $INST" || inst=
[ -n "$RISK" ] && risk="-risk $RISK" || risk=
[ -n "$STACK" -a "$STACK" == "1" ] && stack="-stack" || stack=
[ -n "$DBG" ] && debug="-dbg 3" || debug=

ho.py -i $TIMEFRAME \
    -b \
    -lo $LOSSF \
    -alg $ALGO \
    -sgn $SPACES \
    -pa $PAIRS \
    $risk \
    -amt $AMT \
    $stack \
    -mt ${MT:-1} \
    -mx ${MX:-1} \
    -pts ${PTS:-1} \
    -rpt ${RPT:-10} \
    -smp ${SMP:-1} \
    -j $JOBS -res \
    -e $EPOCHS \
    -mode $MODE \
    $TDATE \
    $inst \
    $debug $@
