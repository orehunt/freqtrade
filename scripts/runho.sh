#!/usr/bin/env bash
set -euo pipefail

[ -n "$INST" ] && inst="-inst $INST" || inst=

ho.py -i 1h \
    -b \
    -lo $LOSSF \
    -alg $ALGO \
    -sgn $SPACES \
    -amt on:roi,trailing,stoploss \
    ${STACKING:--ns} \
    -mt 1 \
    -mx 1 \
    -pts 1 \
    -rpt 10 \
    -j $JOBS -res \
    -e $EPOCHS \
    -mode $MODE \
    $TDATE \
    $inst $@
