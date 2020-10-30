#!/bin/bash

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/opts.sh

# exec jupyter lab --allow-root --ip 0.0.0.0
exec jupyter notebook \
     --notebook-dir=user_data/notebooks/ \
     --no-browser  \
     --allow-root $@
