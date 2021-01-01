#!/bin/sh

path=user_data/jupyter/runtime/
# fetch the most recent kernel
kernel=$(ls -Art "$path"kernel*.json | tail -n 1)
[ -z "$kernel" ] && echo "no kernel connection found" && exit 1
echo connecting to kernel $kernel
sudo chown 1000:1000 $kernel
exec jupyter-console --existing=$kernel
