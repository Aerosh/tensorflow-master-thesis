import subprocess as sp

alphas = [ 3+(0.2*i) for i in range(6)]
taus = [0.00001, 0.0001, 0.001, 0.01, 0.1]

for alpha in alphas:
    if alpha == 3.0 or alpha == 4.0:
        alpha = int(alpha)
    for tau in taus:
        sp.call("echo python resnet/resnet_main.py --dataset=cifar10 --eval_data_path=/data/ext/cifar-10/test_batch* --batch_size=100 --mode=eval --eval_once --image_size=32 --log_root=/home/wp01/users/salottic/tmp/resnet20/size/resnet_alpha1e-" + str(alpha) +
            "/ --alpha=-" + str(alpha) + " --num_gpus=1 --extended --reg size --model_type=resnet20 --zeroing=" + str(tau) + " >> eval-output.txt", shell=True)
        sp.call("python resnet/resnet_main.py --dataset=cifar10 --eval_data_path=/data/ext/cifar-10/test_batch* --batch_size=100 --mode=eval --eval_once --image_size=32 --log_root=/home/wp01/users/salottic/tmp/resnet20/size/resnet_alpha1e-" + str(alpha) +
            "/ --alpha=-" + str(alpha) + " --num_gpus=1 --extended --reg size --model_type=resnet20 --zeroing=" + str(tau) + " >> eval-output.txt", shell=True)
