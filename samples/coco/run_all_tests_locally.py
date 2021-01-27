from pathlib import Path
import os
from subprocess import run

#
colour_correction_types={'grey':'greyworld'} #'alt':'altitude_corrected'}
distortion_correction_types={'nd':'no_distortion', 'd':'distortion_correction'}
rescaling_types={'not_rescaled':'not_rescaled', 'rescaled':'rescaled', 'res_nn':'rescaled_nn', 'drop_res':'dropped_resolution', 'drop_res_nn':'dropped_resolution_nn', 'drop_res_up':'dropped_resolution_scaledup', 'drop_res_up_nn':'dropped_resolution_scaledup_nn'}
base_dir = Path('/home/jenny/Documents/FK2018/tunasand/05_dive')

for colour_correction in colour_correction_types.keys():
	for distortion_correction in distortion_correction_types.keys():
		for rescaling in rescaling_types.keys():
			for n in range(1):
				output_file = '-'.join([colour_correction, distortion_correction, rescaling, str(n)])
				folder = base_dir / colour_correction_types[colour_correction] / distortion_correction_types[distortion_correction] / rescaling_types[rescaling]
				print(output_file)
				if not os.path.exists(f"./logs/{output_file}"):
					os.makedirs(f"./logs/{output_file}", exist_ok=True)
					os.system(f"python3 job_array_nnet.py {n} {folder}/ {output_file} >> ./logs/{output_file}/{output_file}.txt")
