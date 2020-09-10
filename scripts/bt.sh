#!/bin/bash

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/opts.sh

results_path=${userdir}/backtest_results/
file=${results_path}/${timeframe}.json
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

$main_exec \
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

results_file=$(jq -r .'latest_backtest' <${results_path}/.last_result.json)
mv ${results_path}$results_file ${results_path}/${timeframe}.json
