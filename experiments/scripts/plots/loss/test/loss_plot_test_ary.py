import glob, os, re
import numpy as np

files = glob.glob(os.path.join('s*'))
image_num = 922
file_num = 3 * 8
save_file = 'val_loss.npy'

data = []
idx = 0
for nfil in files:
	idx += image_num
	with open(nfil) as fil:
		lines = fil.readlines()
	for line in lines:
		if line[:4] == 'iter':
			value = re.search('\d.\d\d\d\d',line).group()
			value = float(value)
			data.append(value)
	print (nfil,' ',len(data), ' idx=',idx)
data_ary_ori = np.array(data).reshape(file_num, image_num)
average = []
for row in data_ary_ori:
	ave = np.average(row)
	average.append(ave)
data_ary = np.array(average).reshape(3,8)
print ('data_ary=\n',data_ary)
np.save(save_file,data_ary)


