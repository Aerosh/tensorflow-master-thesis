from sys import argv

fname = argv[1]

with open(fname , 'r') as original: data = original.read()
with open(fname , 'w') as modified: modified.write("Layer_num\tsz\talpha\ttau_0\ttau_e-5\ttau_e-4\ttau_e-3\ttau_e-2\ttau_e-1\n" + data)
