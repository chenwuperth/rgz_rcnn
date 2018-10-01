import glob
import os 

files = glob.glob(os.path.join('D*'))
files.sort()
print ('editing \n',len(files),' of files\n ',files)

for idx,fil in enumerate(files):
	print (fil[:2])
	with open(fil) as data:
		lines = data.readlines()
	new_file = open(fil,'w')
	for i, line in enumerate(lines):
		new_line = line
		if i == 17:
			new_line = '                    --imdb rgz_2017_' + fil[:-3]  + ' \\\n'
		if i == 18:
			new_line = '                     --iters 9220 \\\n'	
		if i == 19:
			if not fil[:2] == 'D1':
				new_line = '                    --cfg $RGZ_RCNN/experiments/cfgs/faster_rcnn_end2end.yml \\\n'
		if i == 21:
			if fil[:2] == 'D1':
				new_line = '                    --weights $RGZ_RCNN/output/faster_rcnn_end2end/rgz_2017_trainD1_264/VGGnet_fast_rcnn-'+fil[3]+'0000 \\\n'
			if fil[:2] == 'D3':
				new_line = '                    --weights $RGZ_RCNN/output/faster_rcnn_end2end/rgz_2017_trainD3_600/VGGnet_fast_rcnn-'+fil[3]+'0000 \\\n'
			if fil[:2] == 'D4': 
				new_line = '                    --weights $RGZ_RCNN/output/faster_rcnn_end2end/rgz_2017_trainD4_600/VGGnet_fast_rcnn-'+fil[3]+'0000 \\\n'
		new_file.write(new_line)
	new_file.close()


