#!/bin/bash

divergences=('MMD' 'TV' 'modified_chi_squared')
acquisitions=('GP-UCB' 'DRBOGeneral' 'DRBOWorstCaseSens')

for divergence in "${divergences[@]}"
do
  for acquisition in "${acquisitions[@]}"
  do
    for seed in {0..10}
    do
      CUDA_VISIBLE_DEVICES=0 python exp.py with rand_func acq_name="$acquisition" divergence="$divergence" seed=$seed
    done
  done
done
