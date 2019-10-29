import argparse
import FK2018

def job_array(array_id, data_path):
    add_freq=0.1
    add_pc_freq=0.5
    multiply_freq=0.1
    multiply_pc_freq=0.5
    snp_freq=0.1
    jpeg_freq=0.1
    gaussian_freq=0.1
    motion_freq=0.1
    contrast_freq=0.1
    affine_freq=0.1
    transform_freq=0.1
    elastic_freq=0.1

    if(array_id==0):
       add_pc_freq=0
    if(array_id==1):
       add_freq=0
    if(array_id==2):
       multiply_pc_freq=0
    if(array_id==3):
       multiply_freq=0
    elif(array_id==4):
       snp_freq=0
    elif(array_id==5):
       jpeg_freq=0
    elif(array_id==6):
       gaussian_freq=0
    elif(array_id==7):
       motion_freq=0
    elif(array_id==8):
       contrast_freq=0
    elif(array_id==9):
       affine_freq=0
    elif(array_id==10):
       transform_freq=0
    elif(array_id==11):
       elastic_freq=0

   
    FK2018.train_nnet(dataset=data_path, add_freq=add_freq, add_pc_freq=add_pc_freq, multiply_freq=multiply_freq, multiply_pc_freq=multiply_pc_freq, snp_freq=snp_freq, jpeg_freq=jpeg_freq, gaussian_freq=gaussian_freq, motion_freq=motion_freq, contrast_freq=contrast_freq, affine_freq=affine_freq, transform_freq=transform_freq, elastic_freq=elastic_freq, log_file=str(array_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pass job array id')
    parser.add_argument('array_id', type=int, help='job array id')
    parser.add_argument('data_path', type=str, help='dataset directory')
    args = parser.parse_args()
    job_array(args.array_id, args.data_path)
