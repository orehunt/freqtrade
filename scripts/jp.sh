#!/bin/bash -li

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/scripts/opts.sh

# cleanup previous connection files
# rm user_data/jupyter/runtime/*

exec jupyter notebook \
     --ConnectionFileMixin.connection_fileUnicode='user_data/jupyter/kernel.json' \
     --notebook-dir=user_data/notebooks/ \
     --no-browser  \
     --allow-root $@
