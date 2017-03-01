import subprocess

taus = [0]+ [10**(-i) for i in range(1,6)]
alphas = [-4]
for alpha in alphas:
    print("python resnet/retrieve_network_infos.py --train_data_path=/mnt/nfs/datasets/imagenet/images/ --extended --reg size --log_root=/mnt/nfs/users/salottic/tmp/resnet18/size/residual/resnet_alpha_1e" + str(alpha) + " --alpha=" + str(alpha) +
                        " --model_type=resnet18 --output_file=resnet18-complexity.txt")
    subprocess.call("python resnet/retrieve_network_infos.py --train_data_path=/mnt/nfs/datasets/imagenet/images/ --extended --reg size --log_root=/mnt/nfs/users/salottic/tmp/resnet18/size/residual/resnet_alpha_1e" + str(alpha) + " --alpha=" + str(alpha) +
                        " --model_type=resnet18 --output_file=resnet18-complexity.txt", shell=True)
