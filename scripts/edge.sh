#!/bin/sh

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/opts.sh

export NUMEXPR_MAX_THREADS=16
export FREQTRADE_NOSS=
export FREQTRADE_USERDIR=$userdir

$main_exec edge \
           -c $strategy \
           -c $live \
           -c $exchange \
           -c $amounts \
           -c $amounts_default \
           -c $amounts_tuned \
           -c $askbid \
           -c $pairlists \
           -c $paths \
           -c $edge \
           --userdir $userdir \
           $dmmp \
           --timerange=$timerange \
           -i $timeframe
