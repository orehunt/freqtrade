#!/bin/bash

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/opts.sh

[ -n "$FQT_mTP" ] && mtp="--min-total-profit $FQT_mTP"
[ -n "$FQT_MTP" ] && Mtp="--max-total-profit $FQT_MTP"
[ -n "$FQT_mTD" ] && mtd="--min-trade-duration $FQT_mTD"
[ -n "$FQT_MTD" ] && Mtd="--max-trade-duration $FQT_MTD"
[ -n "$FQT_mAP" ] && map="--min-avg-profit $FQT_mAP"
[ -n "$FQT_MAP" ] && Map="--max-avg-profit $FQT_MAP"

freqtrade hyperopt-list \
	-c $exchange \
	-c $hyperopt \
	-c $amounts \
	-c <(echo "$instance_json") \
	--print-json \
	$mtp \
	$Mtp \
	$mtd \
	$Mtd \
	$map \
	$Map
