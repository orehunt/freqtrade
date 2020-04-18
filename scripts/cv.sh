#!/bin/bash

OPTS="$(realpath $(dirname $_))"
. ${OPTS}/opts.sh

opt_cfg=$userdir/$dir/hyperopt.json
opt_archive=hyperopt_archive
base_data_path=$userdir/${opt_dir}/hyperopt_results.pickle
targets_data_path=$userdir/${opt_dir}/hyperopt_trg_results.pickle
cv_data_path=$userdir/${opt_dir}/hyperopt_cv_results.pickle

if [ -n "$cross_data" ]; then # -r
    cp $cross_data $base_data_path
elif [ -n "$opt_arc" ]; then # -l
    ls -w 1 user_data/hyperopt_archive/*results* |
        sed -r "s/$userdir\/${opt_archive}\/hyperopt_results_(.*).pickle/\1/"
elif [ -n "$opt_cv" ]; then # -v
    if [ "$opt_cv" = 1 ]; then
        sed -r 's/(hyperopt_cv": )false/\1true/' -i $opt_cfg
        else
        sed -r 's/(hyperopt_cv": )true/\1false/' -i $opt_cfg
    fi
    cat < $opt_cfg
elif [ -n "$opt_save" ]; then
    if ! [ -e "$opt_save" ]; then
            cp ${userdir}/${opt_dir}/hyperopt_results.pickle \
               ${opt_save}
    else
        echo "file $opt_save already exists"
    fi
elif [ -n "$opt_move" ]; then
     if [ "$opt_move" = cv ]; then
         mv $base_data_path $targets_data_path
         mv $cv_data_path $base_data_path
     elif [ "$opt_move" = trg ]; then
         mv $base_data_path $cv_data_path
         mv $targets_data_path $base_data_path
     fi
fi
