#!/bin/bash
datasets=( AWA2 CUB SUN APY )
biases=( 5.56 1.12 1.07 12.00 )
epochs=( 15 80 40 60 )
wd=( 1e-4 1e-4 5e-4 1e-3 )
for idx in 0 1 2 3
do
    dset=${datasets[$idx]}
    bias=${biases[$idx]}
    eps=${epochs[$idx]}
    w=${wd[$idx]}
    for type in threshold logistic percentiles gaussian
    do
        python novelty.py --dataset $dset --type $type
        python ale.py --dataset $dset --mode test --epochs $eps --wd $w --novelty --result-file result-$type.txt
        python novelty.py --dataset $dset --type $type --bias $bias
        python ale.py --dataset $dset --mode test --epochs $eps --wd $w --novelty --result-file result-$type-$bias.txt
    done
done
