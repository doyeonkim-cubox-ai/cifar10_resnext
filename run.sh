#!/bin/bash

#SBATCH --job-name=cifar10_resnext
#SBATCH --output="./logs/resnext"
#SBATCH --nodelist=nv174
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "


echo "Run started at:- "
date

# ex) srun python -m mnist_resnet50.train

# training
cnt=0
while [ 1 == 1 ]
do
  if [ $cnt -eq 5 ]; then
    break
  fi
  echo "Start loop after 5sec"
  sleep 5
  srun python -m cifar10_resnext.train -model wresnet
  srun python -m cifar10_resnext.train -model resnext29_8
  srun python -m cifar10_resnext.train -model resnext29_16
  let cnt++
  echo "Sleep for 5sec"
  sleep 5
done

# testing
#srun python -m cifar10_resnext.test -model wresnet
#srun python -m cifar10_resnext.test -model resnext29_8
#srun python -m cifar10_resnext.test -model resnext29_16




