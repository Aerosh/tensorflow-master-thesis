#!/bin/bash

python produce_dat_file.py $1
python add_header $1_kernel-szs.dat
cp $1-kernel_szs.dat ../latex/data/mnist/
mv $1-kernel_szs.dat results/kernel-szs/
mv $1-complexity.txt results/complexity/
mv $1-eval.txt results/eval/
