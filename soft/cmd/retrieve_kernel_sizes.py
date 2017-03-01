from sys import argv
import numpy as np
import json
TAU_MAPPING_TABLE = {
        'tau_0' : 3,
        'tau_e-5' : 4,
        'tau_e-4' : 5,
        'tau_e-3' : 6,
        'tau_e-2' : 7,
        'tau_e-1' : 8 
        }
        

file_name = argv[1]
alpha = float(argv[2])
tau = argv[3]
it_number = int(argv[4])

size_data = np.genfromtxt(file_name, delimiter='\t')[1:,:]
size_data = size_data[size_data[:,2] == alpha,:].astype(int)
max_layer = int(np.amax(size_data[:,0]))
max_size = int(np.amax(size_data[:,1]))
tau_col = TAU_MAPPING_TABLE[tau]
d = 1
kernel_sizes = []
for i in range(max_layer + 1):
    layer = size_data[size_data[:,0] == i,:]
    layer_kernel_size = []
    for j in range(1,max_size+ 1):
        nb_kernel = layer[j,tau_col]
        if  nb_kernel != 0 :
            layer_kernel_size.append((j,nb_kernel,d))

    if(len(layer_kernel_size) > 0):
        kernel_sizes.append(layer_kernel_size)
        d = np.sum(layer[1:,tau_col])
data = {}
data['arch'] = kernel_sizes
data['num_res_units'] = (len(kernel_sizes)- 1)/2
data['num_sub_units'] = [1]* ((len(kernel_sizes)- 1)/2 + (len(kernel_sizes) - 1)%2)
print data
with open('resnet' + str(len(kernel_sizes) + 1 ) + '-config-iter' + str(it_number +
    1) + '.json', 'w') as datafile :
    json.dump(data,datafile)
print('resnet' + str(len(kernel_sizes) + 1) + '-config-iter' + str(it_number + 1) + ' saved !')
