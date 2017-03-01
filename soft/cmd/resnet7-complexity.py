import subprocess

taus = [0]+ [10**(-i) for i in range(1,6)]
alphas = [0] + [10**(-i) for i in range(1,5)]
for alpha in alphas:
    subprocess.call("python resnet/retrieve_network_infos.py --train_data_path=/mnt/nfs/datasets/imagenet/images/ --extended --reg size --log_root=/mnt/nfs/users/salottic/tmp/resnet8/size/residual/resnet_alpha_" + str(alpha) + " --alpha=" + str(alpha) +
                        " --model_type=resnet8-7 --output_file=resnet7-complexity.txt", shell=True) 
