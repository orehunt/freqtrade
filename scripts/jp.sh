#!/bin/bash -li
# NOTE: Need interactive session for proper repl out format
. scripts/opts.sh

# cleanup previous connection files
# rm user_data/jupyter/runtime/*
     #--ServerApp.iopub_data_rate_limit=2000 \
export JULIA_NUM_THREADS=$(nproc)
exec jupyter lab \
     --ConnectionFileMixin.connection_file='user_data/jupyter/kernel.json' \
     --notebook-dir=user_data/notebooks/ \
     --no-browser \
     --allow-root $@
