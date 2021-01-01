#!/bin/bash
#
prefix=$1
suffix=$2

pref=${prefix//\//\\/}
suf=${suffix//\//\\/}
set -x
for f in $(ls ${prefix}*); do
    num=$(echo $f | sed -E "s/.*${pref}([0-9]*)${suf}.*/\1/") ;
    [ -n "${num}" ] && mv $f "${prefix}${num}${suffix}";
done
