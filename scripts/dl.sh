#!/bin/bash

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/opts.sh

list=pairs
quote=${quote:-usdt}

[ -n "$spread" ] && spread="_${spread#_}"
pairsfile=${pairsfile:-"--pairs-file ${pairs_dir}/${exchange_name}_${quote}${spread}.json"}
pairs=${pairs:-$pairsfile}
# datahandler=parquet
# datahandler=json
# datahandler=hdf5
datahandler=zarr

# prefer timerange over days
if [ -n "$timerange" ]; then
	timespan="--timerange $timerange"
elif [ -n "$days" ]; then
	timespan="--days $days"
else
	timespan=
fi
declare -A timeframes
IFS=,
c=0
for tf in $timeframe; do
	timeframes[$c]=$tf
	c=$((c + 1))
done
unset IFS c
for tf in ${timeframes[@]}; do
	freqtrade download-data \
		--data-format-ohlcv $datahandler \
		--data-format-trades $datahandler \
		-c $exchange \
		-c $amounts \
		--exchange $exchange_name \
		$pairs \
		--userdir $userdir \
		-t $tf \
		$timespan \
		--userdir $userdir \
		$erase \
		$debug \
		$dltype
done
