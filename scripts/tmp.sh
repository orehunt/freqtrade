#!/bin/bash

mkdir /tmp/.freqtrade /tmp/.freqtrade/hyperopt_results \
      /tmp/.freqtrade/data
mkdir user_data/hyperopt_results/trials
cp -a plot_data/data/* /tmp/.freqtrade/data/
