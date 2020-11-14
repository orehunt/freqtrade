#!/bin/bash

only_ohlc=${1}

tmp_data=/tmp/.freqtrade/data
hp_data=/tmp/.freqtrade/hyperopt_data
mkdir -p /tmp/.freqtrade $tmp_data $hp_data
mkdir -p user_data/hyperopt_results/trials

cp -a plot_data/data/* $tmp_data
if [ -n "$only_ohlc" ]; then
	exit
fi
cp -a hyperopt_archive/data/* $tmp_data
if [ -d user_data/data ]; then
	rmdir user_data/data || { echo "data dir existing and not empty!, remove"; }
fi

ln -srf $tmp_data user_data/data
if [ -e  user_data/hyperopt_data ]; then
	rmdir user_data/data || { echo "hyperopt_data dir existing and not empty!, remove"; }
fi
ln -srf $hp_data user_data/hyperopt_data
