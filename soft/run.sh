#!/bin/bash

print_command(){
  # directories
  tmp="/home/wp01/users/salottic/tmp/"
  data_root="/data/ext/"
  # Run type
  if [ $1 = "train"  ]
  then
    run_type="models/resnet_main.py "
  elif [ $1 = "eval" ]
  then
    run_type="models/resnet_main.py --mode eval"
  elif [ $1 = "complexity" ]
  then
    run_type="models/retrieve_network_infos.py"
  fi
  # Dataset determination
  dataset=$2
  if [ $dataset = "mnist" ]
  then
    data_folder="MNIST"
  elif [ $dataset="cifar10" ]
  then
    if [ $1 = "train" ]
    then
      data-folder='cifar-10/data_batch_*'
    elif [ $1 = "eval" ]
    then
      data_folder="cifar-10/test_batch.bin"
    else
      data-folder='cifar-10/data_batch_*'
    fi
  fi

  model_type=$3
  optimizer=$4
  reg=$5
  from_alpha=$6
  to_alpha=$7
  step_alpha=$8
  alphas=`seq $from_alpha $step_alpha $to_alpha|xargs`

  comm="python $run_type --dataset $dataset --data_path $data_root$data_folder --log_root $tmp$model_type/$reg-$optimizer/alpha_1e-\$i --alpha  -\$i --optimizer $optimizer --model_type $model_type --reg $reg "

  for_loop="for i in $alphas; do"
  header=$model_type-$reg-$optimizer

  if [ $1 = "train" ]
  then
    comm_fin="$comm --lrn_rate_thrs 0$'\\t'20000$'\\t'30000$'\\t'40000 --lrn_rate_vals 0.001$'\\t'0.0001$'\t'0.00001"
  elif [ $1 = "eval" ]
  then
    for_loop="$for_loop for j in 0 0.00001 0.0001 0.001 0.01 0.1; do"
    comm_fin="$comm --zeroing \$j --eval_output_file $header-eval.txt --eval_batch_count 100 --batch_size 100 --eval_once; done"
  elif [ $1 = "complexity" ]
  then
    comm_fin="$comm --output_file $header"
  else 
    comm_fin="Error - mode doesn't exist"
  fi

  echo "$for_loop $comm_fin; done"
}

print_help(){
  echo "run.sh run_type dataset model_type optimizer regularization alpha_form alpha_to alpha_step"
}

if [ $1 = "--help" ]
then
  print_help
else
  print_command $1 $2 $3 $4 $5 $6 $7 $8
fi

