import csv
from sys import argv

prefix = argv[1]
dat_folder = '../latex/data/'

# Store complexity 

def csv_reader(fname):
    with open(fname,'rb') as comp:
        stack = []
        reader = csv.reader(comp, delimiter='\t')
        for row in reader :
            stack.append(row)

    return stack


complexity_data = csv_reader(prefix + '-complexity.txt')
eval_data = csv_reader(prefix + '-eval.txt')
with open(dat_folder + prefix + '-eval.dat', 'wb') as datfile:
    writer = csv.writer(datfile, delimiter = '\t')
    writer.writerow(['alpha','zeroing','top1','top5','FLOPS'])
    for i in range(len(eval_data)):
        writer.writerow(eval_data[i] + [complexity_data[i][2]])

