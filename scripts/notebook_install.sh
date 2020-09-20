#!/usr/bin/env bash
set -euo pipefail

if [ -e freqtrade ]; then
	echo freqtrade dir already exists
	exit
fi
git clone https://untoreh:RU8pUR7Qq7NuFhucxegS@bitbucket.org/untoreh/freqtrade freqtrade
