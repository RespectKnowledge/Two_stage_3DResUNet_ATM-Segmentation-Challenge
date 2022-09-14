
# #%%
# from pathlib import Path
# import os
# import numpy as np
# import nibabel as nib
# import SimpleITK as sitk
# import glob
# input_folder='C:\\Users\\Administrateur\\Desktop\\mmchalleneges2022\\Parse2021\\docker_submission\\inputs'
# patients = os.listdir(f'{input_folder}')
# for i in patients:
#     print(i)
#     pathim=os.path.join(input_folder,i)
#     #print(pathim)
#     #ImagePath=glob.glob(os.path.join(pathim, '*.nii.gz'))
#     sub = os.path.split(pathim)[1].split('.')[0] # to split the input directory and to obtain the suject name
#     print(sub)
#     Image_bj = sitk.ReadImage(pathim)
    
#     sitk.WriteImage(Image_bj,os.path.join(input_folder,sub+'_0000.nii.gz'))

# #%
# parser = argparse.ArgumentParser(
#         description='lung segmentation of a ct volume')
#     parser.add_argument('--input_dir', default='', type=str, metavar='PATH',
#                         help='this directory contains all test samples(ct volumes)')
#     parser.add_argument('--predict_dir', default='', type=str, metavar='PATH',
#                         help='segmentation file of each test sample should be stored in the directory')

from pathlib import Path
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import glob
import argparse, os
if __name__ == '__main__':
    """
    Prediction for team Dolphins. We trained nnunet with psedo labels
    """
    #
    #inputDir = '/content/drive/MyDrive/Feta2022/validation/sub-012'
    #temp_input='/content/drive/MyDrive/Feta2022/input_temp'
    #output_folder = '/content/drive/MyDrive/Feta2022/output'
    #parameter_folder = '/content/drive/MyDrive/Feta2022/parameters'
    
    #input_folder = './input'
    #output_folder = './predict'
    # parameter_folder = '/workspace/parameters'
    # parser = argparse.ArgumentParser(description='lung segmentation of a ct volume')
    # parser.add_argument('--input_dir', default='', type=str, metavar='PATH',
    #                         help='this directory contains all test samples(ct volumes)')
    # parser.add_argument('--predict_dir', default='', type=str, metavar='PATH',
    #                         help='segmentation file of each test sample should be stored in the directory')

    #args = parser.parse_args()
    #input_dir =  args.input_dir
    #predict_dir = args.predict_dir
    input_dir='/workspace/inputs/'
    predict_dir='/workspace/outputs/'
    parameter_folder = '/workspace/codes/'
    
    
    #input_folder = '/content/drive/MyDrive/Feta2022/validation/'
    #temp_input='/content/drive/MyDrive/Feta2022/input_temp'
    #output_folder = '/content/drive/MyDrive/Feta2022/output'
    #parameter_folder = '/content/drive/MyDrive/Feta2022/parameters'
    #T2wImagePath = glob.glob(os.path.join(inputDir, 'anat', '*_T2w.nii.gz'))[0]
    #sub = os.path.split(T2wImagePath)[1].split('_')[0] # to split the input directory and to obtain the suject name
    #T2wImage = sitk.ReadImage(T2wImagePath)
    #sitk.WriteImage(T2wImage,os.path.join(temp_input,sub+'_0000.nii.gz'))
    # patients = os.listdir(f'{input_dir}')
    # for i in patients:
    #     #print(i)
    #     pathim=os.path.join(input_dir,i)
    #     #print(pathim)
    #     #ImagePath=glob.glob(os.path.join(pathim, '*.nii.gz'))
    #     sub = os.path.split(pathim)[1].split('.')[0] # to split the input directory and to obtain the suject name
    #     #print(sub)
    #     Image_bj = sitk.ReadImage(pathim)
    
    #     sitk.WriteImage(Image_bj,os.path.join(input_dir,sub+'_0000.nii.gz'))

    
    from nnunet.inference.predict import predict_cases
    from batchgenerators.utilities.file_and_folder_operations import subfiles, join
    
    input_files = subfiles(input_dir, suffix='.nii.gz', join=False)
    #output_files = [join(output_folder, i) for i in input_files]
    output_files = [join(predict_dir, i) for i in input_files]
    input_files = [join(input_dir, i) for i in input_files]

    # in the parameters folder are five models (fold_X) traines as a cross-validation. We use them as an ensemble for
    # prediction
    folds = (0)

    # setting this to True will make nnU-Net use test time augmentation in the form of mirroring along all axes. This
    # will increase inference time a lot at small gain, so you can turn that off
    do_tta = False

    # does inference with mixed precision. Same output, twice the speed on Turing and newer. It's free lunch!
    mixed_precision = False

    predict_cases(parameter_folder, [[i] for i in input_files], output_files, folds, save_npz=False,
                  num_threads_preprocessing=1, num_threads_nifti_save=1, segs_from_prev_stage=None, do_tta=do_tta,
                  mixed_precision=mixed_precision, overwrite_existing=True, all_in_gpu=True, step_size=0.5)
    # done!
