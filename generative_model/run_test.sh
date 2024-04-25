#!/usr/bin/env bash

#!/bin/bash
#SBATCH --gres=gpu:1    # Request GPU "generic resources"
#SBATCH --mem=16G  # Requests 4 GB of memory
#SBATCH --account=def-chgag196
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=cyyba@ulaval.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# exit when any command fails
set -e

PROJECT_NAME="def-chgag196"

HOME_PATH="/home/cgarciay"
DATA_DIR=${HOME_PATH}/scratch/data


USERNAME="cgarciay"
REPO_NAME=synthcity
INITIAL_REPO_DIR="/scratch/$USERNAME/$REPO_NAME"
REPO_DIR="$SLURM_TMPDIR/$REPO_NAME"
VENV_DIR="$SLURM_TMPDIR/$REPO_NAME-venv"

##################################
### Load code and dependencies ###



echo "Setting up environment variables..."


#REPO_NAME=Synthetic-Data-Deep-Learning
#INITIAL_REPO_DIR="$REPO_NAME"
#REPO_DIR="$REPO_NAME"
#VENV_DIR="$REPO_NAME-venv"

##################################
### Load code and dependencies ###
##################################
echo "Loading code and dependencies..."

#module load python/3.10.13
#module load python/3.9
#module load scipy-stack

echo "Activating virtual environment..."

echo "Loading code and dependencies..."

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 python/3.9 arrow/12.0.1



#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate
#pip install --no-index --upgrade pip
#pip install --no-index "compute_canada_dependencies/synthcity-0.2.9+computecanada-py3-none-any.whl"
#pip install --no-index -r requirements.txt




ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
#pip install --no-index --upgrade pip
free -h
#
#pip install  synthcity 
pip install --no-index synthcity sklearn numpy pandas==1.5.3
#pip install -no-index numpy==1.23.1
#pip install -no-index  be-great==0.0.5
#pip install -no-index  pandas==1.5.0

#pip install --no-index synthcity
#pip download --no-deps synthcity

#python setup.py install 
pip freeze --local > requirements.txt
python "test.py" 
deactivate
rm -rf $ENVDI





#######################
###  Copy datasets  ###
#######################

# rclone copy "/scratch/$USERNAME/compositing-outputs/rendered_crops" "$SLURM_TMPDIR/rendered_crops" --multi-thread-streams=$SLURM_CPUS_ON_NODE --transfers=100

echo "Starting training"

export PYTHONUNBUFFERED=1 # make sure to print all to logs
#cd $REPO_DIR

                                                                                                                                                                                                                                                                 