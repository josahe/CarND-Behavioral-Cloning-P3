import csv
from shutil import copyfile

path_in = '/mnt/c/Users/joeher01/Projects/car-sim-data/'
path_out = './reduced_data/'

lines=[]
lines_original=[]
with open(path_in+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    i=0
    for line in reader:
        lines_original.append(line)
        if line[3] != '0':
            lines.append(line)
        else:
            if i%5 == 0:
                lines.append(line)
            i+=1     

with open(path_out+'driving_log.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(lines)

for line in lines[1:]:
    for i in range(3):
        path1 = path_in+line[i].strip()
        path2 = path_out+line[i].strip()
        try:
            copyfile(path1, path2)
        except FileNotFoundError as e:
            print(e)

