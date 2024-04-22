import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
import nibabel as nb

from skimage.draw import polygon
from scipy.spatial import ConvexHull
from scipy import ndimage
from skimage.draw import polygon2mask
from skimage.draw import polygon2mask
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splprep
import random
import nrrd

import PSF_generation.python.functions as ff

def find_patches(img, seg = None, patch_size = 256):
    '''img is the original breast image, seg is the segmentation of the blurry region (in the real blur, made by expert), = None for simulation'''
    img_binary = np.zeros_like(img)
    img_binary[img > 0] = 1

    if seg is not None:
        seg[img<=0] = 0

    while True:
        if seg is not None:
            random_pixel = np.where((img_binary == 1) & (seg > 0))
        else:
            random_pixel = np.where(img_binary == 1)
        # randomly pick one pixel
        random_pixel = random.choice(list(zip(random_pixel[0], random_pixel[1])))
        img_patch = img[random_pixel[0]-patch_size//2:random_pixel[0]+patch_size//2, random_pixel[1]-patch_size//2:random_pixel[1]+patch_size//2]
        img_binary_patch = img_binary[random_pixel[0]-patch_size//2:random_pixel[0]+patch_size//2, random_pixel[1]-patch_size//2:random_pixel[1]+patch_size//2]
        if seg is not None:
            seg_patch = seg[random_pixel[0]-patch_size//2:random_pixel[0]+patch_size//2, random_pixel[1]-patch_size//2:random_pixel[1]+patch_size//2]

        # make sure all the pixels in the img_binary_patch are 1
        if seg is not None:
            if np.sum(img_binary_patch) == patch_size**2 and np.sum(seg_patch) == patch_size**2:
                patch_coordinate = [random_pixel[0] - patch_size//2, random_pixel[0] + patch_size//2, random_pixel[1] - patch_size//2, random_pixel[1] + patch_size//2]
                print('patch coordinate: ', random_pixel[0] - patch_size//2, random_pixel[0] + patch_size//2, random_pixel[1] - patch_size//2, random_pixel[1] + patch_size//2)
                break
        else:
            if np.sum(img_binary_patch) == patch_size**2:
                patch_coordinate = [random_pixel[0] - patch_size//2, random_pixel[0] + patch_size//2, random_pixel[1] - patch_size//2, random_pixel[1] + patch_size//2]
                print('patch coordinate: ', random_pixel[0] - patch_size//2, random_pixel[0] + patch_size//2, random_pixel[1] - patch_size//2, random_pixel[1] + patch_size//2)
                break
    return img_patch, patch_coordinate

        

# parameters
PSFsize_list = [[15,15],[22,22],[35,35]]
anxiety = 0.05
numT = 2000
limited_displacement_list = [0.1,0.25,0.5,1.0, 1.5]
MaxTotalLength_range = [2.5,15]
ROI_frequency = 0.5

patches = True
ROI_frequency = [-1 if patches == True else ROI_frequency][0]
patch_size = 256


sheet = '/mnt/BPM_NAS/BPM/alldata/phase3_ge_origin/darwin/cleaned_labels/20240406_blur/all/Blur.csv'
sheet = pd.read_csv(sheet)
# only keep the one without blur
sheet = sheet[(sheet['Blur'] == 1) & (sheet['Exclude'] ==0)]
print('Number of images:', sheet.shape[0])

# for simulation
# result = []
# for index in range(0, sheet.shape[0]):
#     patient = sheet.iloc[index]
#     patient_image_name = patient['Image']
#     print('this file name is: ', patient_image_name)
#     patient_file = os.path.join('/mnt/BPM_NAS/BPM/alldata/phase3_ge_origin/darwin', patient['Folder'], 'data_lut', patient_image_name)

#     dicom_image = pydicom.dcmread(patient_file)

#     img = dicom_image.pixel_array
#     print('shape of the image data:', img.shape)

#     pixel_spacing = dicom_image.ImagerPixelSpacing
#     assert pixel_spacing[0] == pixel_spacing[1]
#     print('Pixel size:', pixel_spacing)

#     img_binary = np.zeros_like(img)
#     img_binary[img > 0] = 1
#     center_of_mass = ndimage.measurements.center_of_mass(img > 0)
#     center_of_mass = [int(center_of_mass[0]), int(center_of_mass[1])]

#     save_folder_main = os.path.join('/mnt/BPM_NAS/BPM/alldata/phase3_ge_origin/darwin/simulations_v3_patches', patient_image_name)
#     ff.make_folder([save_folder_main])

#     # create simulations
#     for random_i in range(0,3):

#         ROI_use = False

#         if random_i == 0:
#             save_folder = os.path.join(save_folder_main, 'static')
#         else:
#             save_folder = os.path.join(save_folder_main, 'sim_'+str(random_i))
#         ff.make_folder([save_folder])

#         if os.path.isfile(os.path.join(save_folder, 'img.nii.gz')) == 1:
#             print('already exist')
#             continue

#         if random_i == 0:
#             if patches == False:
#                 nb.save(nb.Nifti1Image(img, np.eye(4)), os.path.join(save_folder, 'img.nii.gz'))
#             if patches == True:
#                 img_patch, patch_coordinate = find_patches(img, patch_size = patch_size)
                
#                 nb.save(nb.Nifti1Image(img_patch, np.eye(4)), os.path.join(save_folder, 'img.nii.gz'))
#                 nb.save(nb.Nifti1Image(img, np.eye(4)), os.path.join(save_folder, 'img_complete.nii.gz'))
#             continue

#         # Randomly pick one value from PSFsize_list
#         PSFsize = random.choice(PSFsize_list)
#         print("Randomly picked PSFsize:", PSFsize)

#         # Randomly pick one value from limited_displacement_list
#         limited_displacement = random.choice(limited_displacement_list)
#         print("Randomly picked limited_displacement:", limited_displacement)

#         while True:
#             MaxTotalLength = np.random.uniform(MaxTotalLength_range[0], MaxTotalLength_range[1]) 
#             if limited_displacement >= 1 and MaxTotalLength >= 10:
#                 continue
#             else:
#                 break
#         print("Randomly picked MaxTotalLength:", MaxTotalLength)

#         limited_displacement = limited_displacement / pixel_spacing[0]
#         MaxTotalLength = MaxTotalLength / pixel_spacing[0]
                
#         exposure_time = [1]

#         # create motion trajectory
#         TrajCurve = ff.create_motion_trajectory(PSFsize, anxiety, numT, MaxTotalLength,limited_displacement, plot_traj=False)

#         # create PSF for each exposure time
#         PSFS = ff.create_PSF(TrajCurve, exposure_time, PSFsize, plot_PSF=False)

#         # ROI
#         if np.random.uniform(0,1) < ROI_frequency:
#             print('use ROI')
#             ROI = ff.create_random_ROI(img, radius_range = [img.shape[0]//5, img.shape[0]//3],center_of_mass = center_of_mass, plot_ROI=False)
#             ROI_copy = np.copy(ROI)
#             smooth_ROI = cv2.GaussianBlur(ROI_copy.astype(float), (201,201),500)
#             ROI_use = True
#         else:
#             print('no ROI')
#             ROI = np.ones_like(img)
#             smooth_ROI = np.ones_like(img)
#             ROI_use = False
                

#         # create final blurred image
#         final_img = ff.create_motion_blur_img(img, PSFS, add_noise=False, sigma_gauss=0.05)[:,:,0]
#         # scale the final image so that keep the intensity range same
#         final_img = (final_img - np.min(final_img)) / (np.max(final_img) - np.min(final_img)) * (np.max(img) - np.min(img)) + np.min(img)

#         # some chances that only part of the image is blurred
#         if ROI_use:
#             # use smooth ROI
#             final_img_roi = final_img * smooth_ROI + img * (1 - smooth_ROI)
#         else:
#             final_img_roi = final_img

#         # make sure the background is always 0
#         final_img[img<=0] = img[img<=0]; final_img = np.round(final_img)
#         final_img_roi[img<=0] = img[img<=0]; final_img_roi = np.round(final_img_roi)

#         if patches == False:
#             # save the image and ROI
#             nb.save(nb.Nifti1Image(final_img_roi, np.eye(4)), os.path.join(save_folder, 'img.nii.gz'))
#             nb.save(nb.Nifti1Image(final_img, np.eye(4)), os.path.join(save_folder, 'img_entireblur_as_reference.nii.gz'))
#             ROI = ROI.astype(np.uint8)
#             nb.save(nb.Nifti1Image(ROI, np.eye(4)), os.path.join(save_folder, 'ROI.nii.gz'))
#             nb.save(nb.Nifti1Image(smooth_ROI, np.eye(4)), os.path.join(save_folder, 'smooth_ROI.nii.gz'))

#         if patches:
#             final_img_patch, patch_coordinate = find_patches(final_img, patch_size = patch_size)

#             nb.save(nb.Nifti1Image(final_img_patch, np.eye(4)), os.path.join(save_folder, 'img.nii.gz'))
#             nb.save(nb.Nifti1Image(final_img, np.eye(4)), os.path.join(save_folder, 'img_complete.nii.gz')) 
#         # save info
#         if patches == False:
#             result.append([patient['Image'], random_i, ROI_use, PSFsize[0], limited_displacement * pixel_spacing[0], MaxTotalLength * pixel_spacing[0], patient['Blur'], patient['Agree'], patient['PoorQuality'],patient['AdjuDisagree'], patient['Folder'], patient['MRN'], patient['SegFilename'], patient['Dataset']])
#             df = pd.DataFrame(result, columns = ['Image', 'simulation', 'use_ROI?', 'PSFsize', 'limited_displacement(mm)', 'MaxTotalLength(mm)', 'Blur_in_original_image', 'Agree', 'PoorQuality', 'AdjuDisagree', 'Folder', 'MRN', 'SegFilename', 'Dataset'])
#         else:
#             result.append([patient['Image'], random_i, ROI_use, PSFsize[0], limited_displacement * pixel_spacing[0], MaxTotalLength * pixel_spacing[0], patch_coordinate[0],patch_coordinate[1],patch_coordinate[2],patch_coordinate[3], patient['Blur'], patient['Agree'], patient['PoorQuality'],patient['AdjuDisagree'], patient['Folder'], patient['MRN'], patient['SegFilename'], patient['Dataset']])
#             df = pd.DataFrame(result, columns = ['Image', 'simulation', 'use_ROI?', 'PSFsize', 'limited_displacement(mm)', 'MaxTotalLength(mm)', 'patch_coordinate_x_min', 'patch_coordinate_x_max', 'patch_coordinate_y_min', 'patch_coordinate_y_max', 'Blur_in_original_image', 'Agree', 'PoorQuality', 'AdjuDisagree', 'Folder', 'MRN', 'SegFilename', 'Dataset'])
       
#         df.to_excel(os.path.join(os.path.dirname(save_folder_main), 'simulation_info_random.xlsx'), index = False)


# for blur: jus read dicom and save as nii
result = []
for index in range(0, sheet.shape[0]):
    patient = sheet.iloc[index]
    patient_image_name = patient['Image']
    print('this file name is: ', patient_image_name)
    print('patient folder:', patient['Folder'] )
    patient_file = os.path.join('/mnt/BPM_NAS/BPM/alldata/phase3_ge_origin/darwin', patient['Folder'], 'data_lut', patient_image_name)

    dicom_image = pydicom.dcmread(patient_file)

    img = dicom_image.pixel_array
    print('shape of the image data:', img.shape)

    pixel_spacing = dicom_image.ImagerPixelSpacing
    assert pixel_spacing[0] == pixel_spacing[1]
    print('Pixel size:', pixel_spacing)

    img_binary = np.zeros_like(img)
    img_binary[img > 0] = 1
    center_of_mass = ndimage.measurements.center_of_mass(img > 0)
    center_of_mass = [int(center_of_mass[0]), int(center_of_mass[1])]

    save_folder = os.path.join('/mnt/BPM_NAS/BPM/alldata/phase3_ge_origin/darwin/real_blurs_patches', patient_image_name,'original_img')
    ff.make_folder([os.path.dirname(save_folder), save_folder])
    
    seg = os.path.join('/mnt/BPM_NAS/BPM/alldata/phase3_ge_origin/darwin', patient['Folder'], 'seg_blur', patient_image_name + '.Janice.seg.nrrd')
    seg, _ = nrrd.read(seg)
    seg = np.transpose(seg, (1, 0))
    seg[img<=0] = 0
    print('the shape of image and seg:', seg.shape, img.shape)
    nb.save(nb.Nifti1Image(seg, np.eye(4)), os.path.join(save_folder, 'seg.nii.gz'))
    
    if patches == False:
        nb.save(nb.Nifti1Image(img, np.eye(4)), os.path.join(save_folder, 'img.nii.gz'))
    else:
        img_patch, patch_coordinate = find_patches(img, seg, patch_size = patch_size) 
        nb.save(nb.Nifti1Image(img_patch, np.eye(4)), os.path.join(save_folder, 'img.nii.gz'))
        nb.save(nb.Nifti1Image(img, np.eye(4)), os.path.join(save_folder, 'img_complete.nii.gz'))


    if patches == False:
        result.append([patient['Image'],  patient['Blur'], patient['Agree'], patient['PoorQuality'],patient['AdjuDisagree'], patient['Folder'], patient['MRN'], patient['SegFilename'], patient['Dataset']])
        df = pd.DataFrame(result, columns = ['Image', 'Blur_in_original_image', 'Agree', 'PoorQuality', 'AdjuDisagree', 'Folder', 'MRN', 'SegFilename', 'Dataset'])
    else:
        result.append([patient['Image'], patch_coordinate[0],patch_coordinate[1],patch_coordinate[2],patch_coordinate[3], patient['Blur'], patient['Agree'], patient['PoorQuality'],patient['AdjuDisagree'], patient['Folder'], patient['MRN'], patient['SegFilename'], patient['Dataset']])
        df = pd.DataFrame(result, columns = ['Image', 'patch_coordinate_x_min', 'patch_coordinate_x_max', 'patch_coordinate_y_min', 'patch_coordinate_y_max', 'Blur_in_original_image', 'Agree', 'PoorQuality', 'AdjuDisagree', 'Folder', 'MRN', 'SegFilename', 'Dataset'])

df.to_excel(os.path.join('/mnt/BPM_NAS/BPM/alldata/phase3_ge_origin/darwin/real_blurs_patches', 'blur_info.xlsx'), index = False)
