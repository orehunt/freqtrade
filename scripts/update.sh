#!/bin/bash

set -e
rm -f /freqtrade/user_data/data/.gitkeep
rm -df /freqtrade/user_data/data
git pull
pip install -r requirements.txt
pip install -r requirements-hyperopt.txt
pip install -r requirements-plot.txt
ln -sr /tmp/.freqtrade/data/ user_data/
.env/bin/activate
