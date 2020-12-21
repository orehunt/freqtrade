#!/usr/bin/env bash
export MX MT MODE EPOCHS JOBS LOSSF TIMEFRAME INST TDATE SPACES ALGO RES STACK POINTS RPT
function multi {
    export MODE=multi EPOCHS=10000 RES="-res" TDATE="-g 20200101-20201001" INST=
    JOBS=${JOBS:-4}
    SPACES=${SPACES:-buy}
    TIMEFRAME=${TIMEFRAME:-5m}
    ALGO=${ALGO:-Ax:MOO}
    LOSSF=${LOSSF:-MC_tasp}
    STACK=1
}

function cv {
    export MODE=cv EPOCHS=0 JOBS=10 LOSSF=MCCalmarRatio RES="" TDATE="-g 20201001-20201201" INST=last STACK=
}

function tdate {
   export TDATE="-g 20200101-20201001"
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
   PTS=1 MODE=multi JOBS=4 ALGO=Ax:MOO LOSSF=MC_tasp
}
