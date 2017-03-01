import os 
root_dir = './'

for subdur, dir, files in os.walk(root_dir):
    for file in files:
        if 'kernel_szs' in file:
            with open(file,'r') as original : data = original.read()
            import ipdb
            ipdb.set_trace()
            with open(file,'w') as modified :modified.write("Layer_num\tsz\talpha\ttau_0\ttau_e-5\ttau_e-4\ttau_e-3\ttau_e-2\ttau_e-1\n" + data[1:-1])
