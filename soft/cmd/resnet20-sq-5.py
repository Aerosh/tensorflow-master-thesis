import subprocess as sp

alphas = [-(3 + i*0.2) for i in range(2,6)]
taus = [10**(-i) for i in range(1,6)]

for alpha in alphas :
  sp.call("python resnet/resnet_main.py " \
            "--train_data_path=/data/ext/cifar-10/data_batch_* " \
            "--dataset cifar10 " \
            "--log_root /home/wp01/users/salottic/tmp/resnet20-sq-5/size/alpha_1e" + str(alpha) + "/ " \
            "--num_gpus=1 " \
            "--model_type resnet20-square-5 " \
            "--extended " \
            "--reg size " \
            "--alpha " + str(alpha), shell=True)

for alpha in alphas :
    for tau in taus :
        sp.call("python resnet/resnet_main.py " \
                "--eval_data_path=/data/ext/cifar-10/test_batch.bin " \
                "--log_root /home/wp01/users/salottic/tmp/resnet20-sq-5/size/alpha_1e" + str(alpha) + " " \
                "--mode eval " \
                "--num_gpus 1 " \
                "--model_type resnet20-square-5 " \
                "--extended " \
                "--reg size " \
                "--alpha " + str(alpha) + " " \
                "--zeroing " + str(tau) + " " \
                "--eval_output_file resnet20-sq-5-eval.txt " \
                "--eval_once " \
                "--dataset cifar10 " \
                "--batch_size 200 " \
                "--eval_batch_count 50", shell=True)

for alpha in alphas :
	sp.call("python resnet/retrieve_network_infos.py " \
	      "--train_data_path=/data/ext/cifar-10/data_batch_* " \
	      "--extended " \
	      "--reg size " \
	      "--alpha=" + str(alpha) + " " \
	      "--log_root /home/wp01/users/salottic/tmp/resnet20-sq-5/size/alpha_1e" + str(alpha) +  " " \
	      "--model_type resnet20-square-5 " \
	      "--dataset=cifar10 " \
	      "--output_file=resnet20-sq-5-complexity.txt", shell=True)

