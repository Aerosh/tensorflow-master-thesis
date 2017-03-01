from sys import argv
import csv
import numpy as np

M_TABLE = {
        'top1' : 2,
        'FLOPS' : 4
        }

file_name = argv[1]
alpha = float(argv[2])
tau = float(argv[3])


acc_compl = np.genfromtxt(file_name, delimiter = '\t')[1:,:]
acc_compl = acc_compl[((acc_compl[:,0] == alpha) & (acc_compl[:,1] == tau)), :]
acc_compl = acc_compl[0]
print(file_name + " - alpha : " + str(alpha) + " - tau : " + str(tau) 
        + " - top1 : " + str(acc_compl[M_TABLE['top1']]) + " - FLOPS : " +
        str(acc_compl[M_TABLE['FLOPS']]))

