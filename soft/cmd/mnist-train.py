import subprocess

alphas = [3.4, 3.6, 3.8, 4]

for alpha in alphas:
    print "python resnet/resnet_main.py  --log_root=../tmp/resnet5-square-5/size/resnet_alpha_1e-" + str(alpha) + "/ --dataset=mnist --num_gpus=1 --extended --alpha=-" + str(alpha) + " --reg size --gpu_id=0 --batch_size=100 --model_type=resnet5-" +            "square-7"
    subprocess.call("python resnet/resnet_main.py  --log_root=../tmp/resnet5-square-5/size/resnet_alpha_1e-" + str(alpha) + "/ --dataset=mnist --num_gpus=1 --extended --alpha=-" + str(alpha) + " --reg size --gpu_id=0 --batch_size=100 --model_type=" +
            "resnet5-square-7",shell=True)
