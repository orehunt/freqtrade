#!/bin/sh

for db in user_data/hyperopt_data/trials/*.hdf; do
    ptrepack --complib blosc:zstd \
             --complevel 2 \
             "${db}" "${db}.rpk"
    mv "${db}.rpk" "${db}"
done
