#!/bin/bash -x

target_dir=/tmp/.freqtrade/
link_dir=user_data

# while getopts "io" o; do
#     case "$o" in
#         "i") link=in;;
#         "o") link=out;;
#         *)
#     esac
# done

rm -d $link_dir/data/binance
rm -d $link_dir/data/bittrex
rm -d $link_dir/data

lpaths=(data hyperopt_results backtest_data)
for p in ${lpaths[@]}; do
    rm -f "${link_dir}/${p}"
    ln -sr  "${target_dir}/${p}" "${link_dir}/${p}"
done

