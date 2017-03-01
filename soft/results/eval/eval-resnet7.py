import subprocess

taus = [10**(-i) for i in range(1,6)]
alphas = [0] + [10**(-i) for i in range(1,5)]
for alpha in alphas:
    for tau in taus:
        subprocess.call("echo python resnet/resnet_main.py --mode=eval --eval_data_path=/mnt/nfs/datasets/imagenet/images/ --eval_once --log_root=/mnt/nfs/users/salottic/tmp/resnet8/size/residual/resnet_alpha_" + str(alpha) + "/ --num_gpus=1 --extended --alpha=" + str(alpha) +
                        " --reg size --zeroing=" + str(tau) + " --model_type=resnet8-7 >> eval-resnet8.txt", shell=True)
        subprocess.call("python resnet/resnet_main.py --mode=eval --eval_data_path=/mnt/nfs/datasets/imagenet/images/ --eval_once --log_root=/mnt/nfs/users/salottic/tmp/resnet8/size/residual/resnet_alpha_" + str(alpha) + "/ --num_gpus=1 --extended --alpha=" + str(alpha) +  
                        " --reg size --zeroing=" + str(tau) + " --model_type=resnet8-7 >> eval-resnet8.txt", shell=True)
