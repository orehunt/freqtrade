#!/bin/bash -li

export FQT_PBA=1
export tg_bot=live
exec tr.sh -v ${FQT_RUNMODE:-live} -r binance_live.json



