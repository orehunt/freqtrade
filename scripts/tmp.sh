#!/bin/bash

tmp_data=/tmp/.freqtrade/data
hp_data=/tmp/.freqtrade/hyperopt_data
mkdir /tmp/.freqtrade $tmp_data $hp_data
mkdir user_data/hyperopt_results/trials

cp -a plot_data/data/* $tmp_data
cp -a hyperopt_archive/data/* $tmp_data
if [ -d user_data/data ]; then
	rmdir -d user_data/data || { echo "data dir existing and not empty!, remove"; }
fi
ln -srf $tmp_data user_data/data
ln -srf $hp_data user_data/hyperopt_data
