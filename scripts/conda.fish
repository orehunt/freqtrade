#!/usr/bin/fish

set conda_path .env/miniconda3/bin
set conda_bin .env/miniconda3/bin/conda

set -gx PATH $PATH $conda_path
eval $conda_bin "shell.fish" "hook" $argv | source
