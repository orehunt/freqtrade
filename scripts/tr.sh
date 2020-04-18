#!/bin/bash

[ -n "$BASH_SOURCE" ] && src=$BASH_SOURCE || src=$_
OPTS="$(realpath $(dirname $src))"
. ${OPTS}/opts.sh

if [ "$runmode" = live ]; then
    tg_bot=${tg_bot:-live}
    dburl=${dburl:-"sqlite:///live.sqlite"}
    dryrun_flag=false
else
    tg_bot=${tg_bot:-dry}
    dburl=${dburl:-"sqlite:///dry.sqlite"}
    dryrun_flag=true
fi

telegram=$dir/tg/${exchange_name}_${tg_bot}_telegram.json
exchange=$dir/binance.json
amounts=$dir/amounts.json

if [ "$open_trades" = "-1" ]; then
    edge=$dir/edge.json
else
    edge=
fi

config_files="$(echo $strategy $live $askbid $amounts $amounts_tuned \
                     $edge $pairlists $exchange $paths $telegram)"

[ -n "$edge" ] && echo "EDGE is Enabled!"

[ "$runmode" != live ] && dryrun="--dry-run"
if [ -n "$config" ]; then
    if jq -rR 'select( test("\\s*(//|/\\*)")|not)'  \
          $config_files |  \
            jq -s "add | .dry_run = $dryrun_flag" \
               > ${exchange_name}_${runmode}.json; then
        echo "Runtime config was saved at ${exchange_name}_${runmode}.json"
    else
            echo "Error: check json syntax of config files"
    fi
else
    if [ -n "$runtime_config" ]; then
        pgrep syslogd || syslogd -t &>/dev/null
        exec freqtrade trade \
             -c $runtime_config \
             --db-url $dburl \
             --logfile syslog:/dev/log \
             $dryrun \
             ${debug}
    else
        echo "runtime_config not specified (-r config.json)"
        exit 1
    fi
fi


