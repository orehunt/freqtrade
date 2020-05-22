#!/bin/bash

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/opts.sh

list=pairs
quote=${quote:-usdt}

[ -n "$spread" ] && spread="_${spread#_}"
pairsfile=${pairsfile:-"--pairs-file ${pairs_dir}/${exchange_name}_${quote}${spread}.json"}
pairs=${pairs:-$pairsfile}

freqtrade download-data \
          --data-format-ohlcv parquet \
          -c $exchange \
          -c $amounts \
          --exchange $exchange_name \
          $pairs \
	        --userdir $userdir \
          -t $timeframe \
          --days $days \
          --userdir $userdir
          $dltype

