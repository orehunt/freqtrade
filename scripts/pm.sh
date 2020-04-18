#!/bin/bash

. .env/bin/activate

if [ -n "$1" ]; then
    freqtrade hyperopt-show -n $1 --print-json 2>/dev/null
    exit
fi

declare -i c
while read n; do
  c+=1
  freqtrade hyperopt-show --best -n $c --no-header 2>/dev/null
done < <(freqtrade hyperopt-list --best --no-details --print-json 2>/dev/null)
