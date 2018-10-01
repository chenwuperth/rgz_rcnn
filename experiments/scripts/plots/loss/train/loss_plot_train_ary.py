import glob, os, re
import numpy as np

files = ['D1_264_train.out','D3_600_train.out','D4_600_train.out']
file_num = len(files)
# iter = 80000 -> iters =8000
iters = 8000
save_file = 'tra_loss.npy'
data_num = 12280
# processing txt data
data = []
for nfil in files:
	with open(nfil) as fil:
		lines = fil.readlines()
	for idx,line in enumerate(lines):
		if line[:4] == 'iter':
			value = re.search('\d.\d\d\d\d',line).group()
			value = float(value)
			if value > 1:
				print ('file: ',nfil,'line: ',str(idx), 'value: ',value)
			data.append(value)
data_ary_ori = np.array(data).reshape(file_num,iters)
index = [ num*1000 for num in range(1,9)]
data_ary = []

for i, row in enumerate(data_ary_ori):
	if i == 0:
		#D1_264 has 6141 data, other has 6140 data.
		data_num = data_num + 1*2
# calculation of loss
	for p in index:
		start_point = p - data_num
		if start_point < 0: #get loss for 10000 iter
			start_point = 0 #get loss for all data (12280) at the point of 20k,30k....80k
		average = np.average(row[start_point:p])
		data_ary.append(average)

data_ary = np.array(data_ary).reshape(file_num,len(index))
np.save(save_file,data_ary)
print (save_file, ' was saved\n',data_ary)

