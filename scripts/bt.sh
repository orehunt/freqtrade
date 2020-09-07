#!/bin/bash

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/opts.sh

file=${userdir}/backtest_results/${timeframe}.json
[ "$(ls ${file%\.json}* 2>/dev/null | wc -l)" -gt 0 ] && rm -f ${file%\.json}*

days=${days:-21}
source=file # db
export=trades
db=
if [ -n "$debug" ]; then
	debug="-$debug"
fi

export NUMEXPR_MAX_THREADS=16
export FREQTRADE_NOSS=
export FREQTRADE_USERDIR=$userdir

pairlists=$dir/pairlists_static.json

exec $main_exec \
	backtesting -c $hyperopt \
	-c $strategy \
	-c $live \
	-c $exchange \
	-c $amounts \
	-c $amounts_default \
	-c $amounts_tuned \
	-c $askbid \
	-c $pairlists \
	-c $paths \
	--userdir $userdir \
	$dmmp \
	$eps \
	$open_trades_arg \
	$stake_amount_arg \
	--timerange "$timerange" \
	--export=$export \
	--export-filename=$file \
	-i $timeframe \
	$debug
