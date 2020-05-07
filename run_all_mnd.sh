#!/bin/bash
python train.py --dataset AWA2 --epochs 20 --mode val --g 0.5
python train.py --dataset AWA2 --epochs 20 --mode test --g 0.5
python train.py --dataset CUB --epochs 25 --mode val --g 0.5
python train.py --dataset CUB --epochs 25 --mode test --g 0.5
python train.py --dataset SUN --epochs 25 --mode val --g 0.5
python train.py --dataset SUN --epochs 25 --mode test --g 0.5
python train.py --dataset APY --epochs 20 --mode val --g 0.5
python train.py --dataset APY --epochs 20 --mode test --g 0.5
