#!/bin/bash

src_dir=/tmp/.freqtrade/
dst_dir=user_data
data_dir=plot_data

# while getopts "io" o; do
#     case "$o" in
#         "i") link=in;;
#         "o") link=out;;
#         *)
#     esac
# done

rm -f -d $link_dir/data/binance
rm -f -d $link_dir/data/bittrex
rm -f -d $link_dir/data

lpaths=(data hyperopt_results hyperopt_data backtest_data)
for p in ${lpaths[@]}; do
    rm -f "${src_dir}/${p}"
    mkdir -p "${src_dir}/${p}"
    ln -sr  "${src_dir}/${p}" "${dst_dir}/${p}"
done

# copy data to tmpfs
for d in ${data_dir}/data/*; do
    name="$(basename $d)"
    mkdir -p "${src_dir}/data/${name}"
    cp -a "${d}/"* "${src_dir}/data/${name}/"
done

# jupyter
rm -f -d ~/.local/share/jupyter
ln -sr /freqtrade/user_data/jupyter ~/.local/share/jupyter
