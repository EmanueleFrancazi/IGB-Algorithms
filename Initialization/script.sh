#!/bin/bash

##SBATCH -A eawag
#SBATCH -n 1
##SBATCH --ntasks-per-node=1
#SBATCH --job-name=PT_ENV2             #This is the name of your job

#SBATCH --cpus-per-task=16                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=8G              #This is the memory reserved per core.
#Total memory reserved: 2GB

#SBATCH --time=120:00:00      #This is the time that your task will run
##SBATCH --time=120:00:00      #This is the time that your task will run
##SBATCH --time=2weeks        #This is the time that your task will run
##SBATCH --qos=1week          #You will run in this queue

#SBATCH --tmp=1G    # the compute node should have at least the selected amount of free space in local scratch folder ($TMPDIR)

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=myrun.o%j     #These are the STDOUT and STDERR files
#SBATCH --error=myrun.e%j


#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:10g


##SBATCH --gpus=1

##SBATCH --partition=v100
##SBATCH --gres=gpu:1


#This job runs from the current working directory


#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $SLURM_JOBID stores the ID number of your job.


#load your required modules below
#################################

#shared module managing (to have access to new python version you have to load also the corresponding gcc version to satisfy all the dependencies)

module load eth_proxy
export http_proxy=http://proxy.ethz.ch:3128

module load gcc/11.4.0 python/3.11.6



#source $HOME/Environments/IGB_env/bin/activate
source $HOME/Environments/IGB_Updated/bin/activate


#export your required environment variables below
#################################################


#Lpwd=$(pwd)

#rsync -a --info=progress2 --no-i-r -hs $Lpwd/data_nobackup $TMPDIR --exclude-from='ExcludedFolders.txt'                  #option added to have percentage of the total process (see https://serverfault.com/questions/219013/showing-total-progress-in-rsync-is-it-possible (also comments))

#rsync -a  $Lpwd/data_nobackup $TMPDIR --exclude-from='ExcludedFolders.txt'              #quick option without information transfer

export WANDB_API_KEY=31c77a8754ddfbb90d84833179286e1eba958727

export WANDB_MODE=offline


#add your command lines below
#############################

nohup ./PythonRunManager.sh 1 6000

deactivate
