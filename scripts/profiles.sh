#!/usr/bin/env bash
export MX MT MODE EPOCHS JOBS LOSSF TIMEFRAME INST TDATE SPACES ALGO RES STACK POINTS RPT AMT DBG RISK PTS

function multi {
	export MODE=multi EPOCHS=10000 RES="-res" TDATE="-g 20200101-20201201" INST=
    JOBS=${JOBS:-4}
    SPACES=${SPACES:-buy}
    TIMEFRAME=${TIMEFRAME:-5m}
    ALGO=${ALGO:-Ax:BOTORCH}
    LOSSF=${LOSSF:-MCCalmarRatio}
    AMT=${AMT:-on\:roi,trailing,stoploss}
    STACK=1
}

function cv {
    export MODE=cv EPOCHS=0 JOBS=10 LOSSF=MCCalmarRatio RES="" TDATE="-g 20201001-20201201" INST=last STACK=
}

function tdate {
   export TDATE="-g 20200101-20201201"
}

function tdate2 {
   export TDATE="-g 20201201-"
}

function tdate0 {
   export TDATE="-g 20200101-20201001"
}

function tdate1 {
   export TDATE="-g 20201001-20201201"
}

function tdatef {
   export TDATE="-g 20200101-"
}

function asha {
   PTS=1 MODE=single ALGO=Sherpa:ASHA JOBS=14 LOSSF=MCCalmarRatio
}

function moo {
   PTS=1 RPT=3 MODE=multi JOBS=4 ALGO=Ax:MOO LOSSF=STCLoss
}

function amoo {
   PTS=1 RPT=3 MODE=multi JOBS=4 ALGO=Ax:auto_moo LOSSF=STCLoss
}

function pbt {
   PTS=8 MODE=single JOBS=8 RPT=8 ALGO=Sherpa:PBT LOSSF=MCCalmarRatio
}

function axbo {
   PTS=1 MODE=multi JOBS=8 RPT=3 ALGO=Ax:BOTORCH LOSSF=MCCalmarRatio
}

function skbo {
   PTS=1 MODE=single JOBS=4 RPT=9 ALGO=Skopt:ET LOSSF=MCCalmarRatio
}
