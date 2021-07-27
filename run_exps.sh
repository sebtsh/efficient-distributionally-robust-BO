#!/bin/bash

divergences=('MMD' 'TV' 'modified_chi_squared')
acquisitions=('GP-UCB' 'DRBOGeneral' 'DRBOWorstCaseSens')

for divergence in "${divergences[@]}"
do
  for acquisition in "${acquisitions[@]}"
  do
    for seed in {0..10}
    do
      echo "Running exp for $divergence $acquisition $seed"
    done
  done
done