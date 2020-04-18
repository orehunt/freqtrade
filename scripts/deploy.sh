#!/bin/sh

repo=github.com/orehunt/freqtrade
git clone --depth=1 https://$repo freqtrade
cd /freqtrade
apk add coreutils
pip3 install cython wheel
pip3 install --no-index --find-links wheels/ .

find ./ -name libta_\* || echo "no talib found"
# pip3 install joblib technical
pip3 install -e .
# ln -sr user_data $fqt_dir
