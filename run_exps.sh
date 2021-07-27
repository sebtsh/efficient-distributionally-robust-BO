#!/bin/bash

divergences=('MMD' 'TV' 'modified_chi_squared')
acquisitions=('GP-UCB' 'DRBOGeneral' 'DRBOWorstCaseSens')

for divergence in "${divergences[@]}"
do
  for acquisition in "${acquisitions[@]}"
  do
    for seed in {0..10}
    do
      if [[ $acquisition = 'DRBOWorstCaseSens' ]]
      then
        beta=0
      else
        beta=2
      fi
      CUDA_VISIBLE_DEVICES=0 nohup python exp.py with rand_func acq_name="$acquisition" divergence="$divergence" seed=$seed beta_const=$beta &>/dev/null &
    done
  done
done
