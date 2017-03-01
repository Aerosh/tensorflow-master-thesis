import csv 
from sys import argv

root = argv[1]

datfile= open('resnet10-width-size-nb-kernel-steps.dat','wb')
datwriter = csv.writer(datfile, delimiter='\t')
datwriter.writerow(['layer_nb', 'alpha', 'tau' , 'nb_kernel', 'step' ])
for layer_nb in [0,1,2,7,8]:
    for alpha in [-1.0, -2.0, -3.0, -4.0]:
        for tau in [0.01, 0.001, 0.0001]:
            print('reading : ' +
                    'run_alpha_1e' + str(alpha) + '-_alpha_' +str(10**alpha) + 
                    ',tag_Layer ' + str(layer_nb) + '-#Kernel - tau '+ str(tau)  +
                    '.csv')
            with open(root + 'run_alpha_1e' + str(alpha) + '-_alpha_' +str(10**alpha) + 
                    ',tag_Layer ' + str(layer_nb) + '-#Kernel - tau '+  str(tau)  +
                    '.csv') as csvfile:

                csvreader = csv.reader(csvfile, delimiter=',')
                for i,row in enumerate(csvreader):
                    if i != 0:
                        datwriter.writerow([layer_nb, alpha, tau, row[2], row[1]])


