#!/usr/bin/env bash

function multi {
    export MODE=multi EPOCHS=10000 JOBS=4 LOSSF=MC_tasp ALGO=Ax:MOO SPACES=buy RES="-res" TDATE="-g 20200101-20201001"
}

function cv {
    export MODE=cv EPOCHS=0 JOBS=10 LOSSF=MCCalmarRatio SPACES=buy RES="" TDATE="-g 20201001-20201201" INST=last
}

function cv2 {
   export TDATE="-g 20201201-"
}
