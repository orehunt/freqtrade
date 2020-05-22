#!/bin/bash

tmp_data=/tmp/.freqtrade/data
mkdir /tmp/.freqtrade $tmp_data
mkdir user_data/hyperopt_results/trials

cp -a plot_data/data/* $tmp_data
ln -srf $tmp_data user_data/data
