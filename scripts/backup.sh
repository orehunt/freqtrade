#!/bin/sh

dir=/tmp/stuff
name=freqtrade.zip
datadir=~/.tmp/freqtrade/user_data
bfile="${dir}/${name}"


mkdir -p $dir
rm "${bfile}"

cp "freqtrade/freqtradebot.py" \
   "freqtrade/optimize/hyperopt.py" \
   "freqtrade/strategy/interface.py" \
   "${datadir}/stash/"

zip "${bfile}" -r \
    "cfg" \
    "$datadir" \
    -x "*/__pycache__/*" \
    -i "*/strategies/*" \
    "*/*.org" \
    "*/*.csv" \
    "*/*.py" \
    "*/freqtrade-strategies*/" \
    "*/hyperopts/*" \
    "*/notebooks/*" \
    "*/pairlists/*" \
    "*/cfg/*" \
    "*/modules/*" \
    "*/indicators/*" \
    "*/params/*" \
    "*/raw/*" \
    "*/archive/*" \
    "*/stash/*" \
    "*/scripts/*" \
    "../params.json*"
