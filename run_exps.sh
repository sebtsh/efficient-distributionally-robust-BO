#!/bin/bash

divergences=('MMD' 'TV' 'modified_chi_squared')
acquisitions=('GP-UCB' 'DRBOGeneral' 'DRBOWorstCaseSens' 'DRBOCubicApprox')
betas=(0 0.5 1 2)
means=(0 0.25 0.5)

for divergence in "${divergences[@]}"
do
  for acquisition in "${acquisitions[@]}"
  do
    for ((seed=$1; seed <= $2; seed++))
    do
      for beta in "${betas[@]}"
      do
        for mean in "${means[@]}"
        do
          CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 nohup python exp.py with rand_func acq_name="$acquisition" divergence="$divergence" seed=$seed beta_const=$beta ref_mean=$mean &>/dev/null &
        done
      done
    done
  done
done
