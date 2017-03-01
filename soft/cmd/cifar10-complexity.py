import subprocess as sp

alphas = [ 3+(0.2*i) for i in range(6)]
taus = [0]

for alpha in alphas:
    if alpha == 3.0 or alpha == 4.0:
        alpha = int(alpha)
    for tau in taus:
        sp.call("echo python resnet/retrieve_network_infos.py --train_data_path=/data/ext/cifar-10/data_batch_* --extended --reg=size --alpha=-" + str(alpha) + " --log_root=../../../../../tmp/resnet20/size/resnet_alpha1e-" + str(alpha) + "/ --model_type=resnet20 --dataset=cifar10 --tau=" + str(tau) + " >> complexity-output.txt", shell=True)
        sp.call("python resnet/retrieve_network_infos.py --train_data_path=/data/ext/cifar-10/data_batch_* --extended --reg=size --alpha=-" + str(alpha) + " --log_root=../../../../../tmp/resnet20/size/resnet_alpha1e-" + str(alpha) + "/ --model_type=resnet20 --dataset=cifar10 --tau=" + str(tau) + " >> complexity-output.txt", shell=True)
