export MX MT MODE EPOCHS JOBS LOSSF TIMEFRAME INST TDATE SPACES ALGO RES STACK POINTS RPT AMT DBG= RISK= PTS SMP PAIRS=

function tr0 {
   export TDATE="-g 20200101-20201201"
}

function tr1 {
   export TDATE="-g 20201201-"
}

function trf {
   export TDATE="-g 20200101-"
}

function defopt {
   JOBS=4 EPOCHS=1000 RES="-res"
   TDATE="$(trf)" INST= TIMEFRAME="5m"
   ALGO=Ax:BOTORCH LOSSF=MCCalmarRatio
   AMT="on:roi,trailing,stoploss"
   STACK= SPACES="buy,roi,trailing,stoploss" PTS=1 SMP=1000
}

function multi {
	MODE=multi JOBS=4 STACK= EPOCHS=10000 RES="-res" TDATE="$(trf)" INST=
}

function cv {
    export MODE=cv EPOCHS=0 JOBS=10 LOSSF=MCCalmarRatio RES="" TDATE="$(tr1)" INST=last STACK=
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
   PTS=1 SMP=100 MODE=multi JOBS=8 RPT=3 ALGO=Ax:BOTORCH LOSSF=MCCalmarRatio
}

function skbo {
   PTS=1 MODE=single JOBS=4 RPT=9 ALGO=Skopt:ET LOSSF=MCCalmarRatio
}
