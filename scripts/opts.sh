#!/bin/bash

while getopts "ta:d:q:e:i:p:f:xm:ctd:s:n:g:u:l:r:y:o:v:b:j" o; do
	case "$o" in
	a) stake_amount="$OPTARG" ;;
	t) dltype="--dl-trades" ;;
	d) days="$OPTARG" ;;
	q) quote="$OPTARG" ;;
	e) exchange_name="$OPTARG" ;;
	i)
		timeframe="$OPTARG"
		instance="$OPTARG"
		;;
	g) timerange="$OPTARG" ;;
	p) pairs="--pairs $OPTARG" ;;
	f)
		pairsfile="--pairs-file $OPTARG"
		profile=$OPTARG
		;;
	u) userdir=$OPTARG ;;
	m)
		dmmp="--dmmp"
		opt_move=$OPTARG
		;;
	o) open_trades=$OPTARG ;;
	s)
		eps="--eps"
		opt_save=$OPTARG
		spread=$OPTARG
		;;
	c)
		clear=true
		config=true
		;;
	n)
		tuned_amounts=$OPTARG
		trial_index=$OPTARG
		;;
	l)
		sqlite=$OPTARG
		opt_arc=1
		;;
	r)
		cross_data=$OPTARG
		plot_type=1
		runtime_config=$OPTARG
		;;
	z) testcond=$OPTARG ;;
	y) strategy_name=$OPTARG ;;
	v)
		runmode=$OPTARG
		opt_cv=$OPTARG
		;;
	b) dburl="--db-url $OPTARG" ;;
	j) debug="-v${OPTARG}" ;;
	x) set -x ;;
	*) ;;
	esac
done

if [ -n "$profile" ]; then
	main_exec="python -m $profile $(which freqtrade)"
else
	main_exec=$(which freqtrade)
fi
timeframe=${timeframe:-1h}
dir=cfg/
opt_dir=hyperopt_results
userdir=${userdir:-user_data}
exchange_name=${exchange_name:-exchange_testing}
exchange_tag=${exchange_name/*_/}
if [ "$exchange_tag" == "$exchange_name" ]; then
	exchange_tag=
else
	exchange_tag=_${exchange_tag}
fi
exchange_name=${exchange_name/_*/}
if [ -n "$strategy_name" ]; then
	strategy=$dir/strategies/${strategy_name}.json
else
	strategy=$dir/strategy.json
fi

hyperopt=$dir/hyperopt.json
edge=$dir/edge_backtesting.json
live=$dir/live.json
askbid=$dir/askbid.json
exchange=${exchange:-$dir/${exchange_name}${exchange_tag}.json}
paths=$dir/paths.json
pairlists=$dir/pairlists.json
pairs_dir=${dir}/pairlists
if [ -e "${dir}/${exchange_name}/amounts_backtesting.json" ]; then
	amounts=$dir/${exchange_name}/amounts_backtesting.json
else
	amounts=$dir/amounts_backtesting.json
fi
amounts_default=$dir/amounts/default.json
amounts_pairs=$dir/amounts/pairs_amounts.json

if [ -n "$tuned_amounts" ]; then
	if [ "$tuned_amounts" = 0 ]; then
		amounts_tuned=$dir/amounts_backtesting_tuned.json
	elif [ "$tuned_amounts" == off ]; then
		amounts_tuned=$dir/amounts/off.json
	elif [ "$tuned_amounts" != "${tuned_amounts#cfg}" ]; then
		amounts_tuned=${tuned_amounts}
	else
		amounts_tuned=$dir/amounts/${tuned_amounts}.json
	fi
else
	amounts_tuned=$dir/roi/${timeframe}.json
fi

# get last trials instance from file
if [ -n "$instance" ]; then
	if [ $instance == "last" ]; then
		instance=$(jq -r '.[-1]' <${userdir}/hyperopt_data/trials_instances.json)
	elif [ $instance == "last:cv" ]; then
		instance=$(jq -r '.[-1]' <${userdir}/hyperopt_data/trials_instances.json)_cv
	fi
	instance_json='{"hyperopt_trials_instance": "'"$instance"'"}'
else
	instance_json="{}"
fi

[ -n "$open_trades" ] && open_trades_arg="--max-open-trades $open_trades"
[ -n "$stake_amount" ] && stake_amount_arg="--stake-amount $stake_amount"

days=${days:-21}
timestamp="$(date +%s)"
timerange=${timerange:-$(echo "scale=0; $timestamp-3600*24*$days" | bc)-}

pickles="${userdir}/backtest_data/*.pickle"
[ -n "$clear" ] && rm -f $pickles

quote=${quote:-USDT}

export NUMEXPR_MAX_THREADS=16
export FQT_NOSS=
export FQT_USERDIR=$userdir
export FQT_TEST_COND=$testcond
