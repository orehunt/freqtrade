#!/bin/sh
## save results

rs_number=${1}
[ -z "$rs_number" ] && { echo no result number specified; exit 1; }

ho_path=/tmp/.freqtrade/hyperopt/
ho_archive=user_data/hyperopt_archive/
mkdir -p "$ho_path"
rm "${ho_path}/hyperopt_"*
ln -srf "${ho_archive}/${rs_number}/hyperopt_"* "${ho_path}"

rs_path=/tmp/.freqtrade/hyperopt/results/$rs_number
freqtrade hyperopt-list \
          --best \
          --no-color \
          --print-json 2>/dev/null | \
    grep -E '^{.*}$' > "$rs_path"
