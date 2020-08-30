#!/bin/bash

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/opts.sh

# indicators1="sb" indicators2="macd"
indicators1="sb" indicators2="liqu illiqu spread"
# indicators1="tn kj sa sb ck sh lo emas emam mp mpp tl ub mb lb"
# indicators1="tn kj sa sb ck ub mb lb"
# indicators1="sh lo emas emam mp mpp tl ub mb lb sar"
# indicators2="macd macds macdh"
# indicators2="cci cmo mdi pdi adx trg mom"
# indicators2="ad "
# indicators2="bop ppo ard aru"
# indicators2="rsi stok stod"
# indicators2="avgp medp typp wclp"
# indicators2="htpe htph htin htqu htsi htls htgr"
# indicators1="t1 t2 t3 t4"
# indicators2="pr t1 t2 t3 t4 adx_max adx_min pdi_max pdi_min mdi_max mdi_min"
# indicators2="buy_guard_false buy_guards_count sell_guard_false sell_guards_count"
# indicators=""

limit=35000
if [ -n "$plot_type" ]; then
    plot_type=plot-profit
else
    plot_type=plot-dataframe
fi
if [ -n "$sqlite" ]; then
    source=DB
    db="--db-url=sqlite:///$sqlite"
    file=
else
    source=file
    file="--export-filename=${userdir}/backtest_results/${timeframe}.json"
    db=
fi

dfArgs="--plot-limit $limit --indicators1 $indicators1 --indicators2 $indicators2"
[ "$plot_type" = "plot-profit" ] && unset dfArgs
export NUMEXPR_MAX_THREADS=16
export FREQTRADE_NOSS=
export FREQTRADE_USERDIR=$userdir
freqtrade $plot_type \
          -c $strategy \
          -c $hyperopt \
          -c $live \
          -c $exchange \
          -c $amounts \
          -c $amounts_default \
          -c $amounts_tuned \
          -c $askbid \
          -c $pairlists \
          -c $paths \
          -i $timeframe \
          --userdir $userdir \
          --trade-source=$source \
          $file \
          $db \
          --timerange=$timerange \
          $dfArgs \
          $pairs
