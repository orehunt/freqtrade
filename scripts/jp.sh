#!/bin/sh

# exec jupyter lab --allow-root --ip 0.0.0.0
exec jupyter notebook \
     --notebook-dir=/var/home/fra/.tmp/freqtrade/user_data/notebooks/ \
     --no-browser  \
     --allow-root $@
