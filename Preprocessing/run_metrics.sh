#!/usr/bin/env bash

#SBATCH --account=def-chgag196
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=cyyba@ulaval.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# exit when any command fails
set -e



echo "Setting up environment variables..."

PROJECT_NAME="def-chgag196"
USERNAME="cgarciay"

REPO_NAME=Synthetic-Data-Deep-Learning
INITIAL_REPO_DIR="$REPO_NAME"
REPO_DIR="$REPO_NAME"
VENV_DIR="$REPO_NAME-venv"

##################################
### Load code and dependencies ###
##################################

echo "Loading code and dependencies..."
module --ignore_cache load python/3.9


# tar -xf "$VENV_ARCHIVE" -C "$VENV_DIR" .

echo "Activating virtual environment..."

# Quite hacky, but overrides venv's bin path if different from login node
# export PATH=$VENV_DIR/bin:$PATH
# echo $PATH
virtualenv --no-download "$VENV_DIR"
source "$VENV_DIR/bin/activate"

#export FFTW=/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx512/Compiler/gcc9/fftw/3.3.8

pip install --no-index -r "requirements.txt"


#######################
###  Copy datasets  ###
#######################

# rclone copy "/scratch/$USERNAME/compositing-outputs/rendered_crops" "$SLURM_TMPDIR/rendered_crops" --multi-thread-streams=$SLURM_CPUS_ON_NODE --transfers=100

echo "Starting training"

export PYTHONUNBUFFERED=1 # make sure to print all to logs
#cd $REPO_DIR
python "input_pred.py" --type_a "outs_visit"                                                                                                                                                                                                                                                                  