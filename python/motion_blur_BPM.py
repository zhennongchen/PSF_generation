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

import PSF_generation.python.functions as ff

# parameters
PSFsize_list = [[35,35],[64,64]]
anxiety = 0.05
numT = 2000
limited_displacement_list = [1.0, 1.5]
MaxTotalLength_range = [20,100]
ROI_frequency = 0.4


sheet = '/mnt/BPM_NAS/cleaned_labels/20240406_blur/all/Blur.csv'
sheet = pd.read_csv(sheet)
# only keep the one without blur
sheet = sheet[(sheet['Blur'] == 0) & (sheet['Exclude'] ==0)]
print('Number of images:', sheet.shape[0])

# for simulation
result = []
for index in range(0, sheet.shape[0]):
    patient = sheet.iloc[index]
    patient_image_name = patient['Image']
    print('this file name is: ', patient_image_name)
    patient_file = os.path.join('/mnt/BPM_NAS', patient['Folder'], 'data_lut', patient_image_name)

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

    save_folder_main = os.path.join('/mnt/BPM_NAS/simulations', patient_image_name)
    ff.make_folder([save_folder_main])

    # create simulations
    for random_i in range(0,4):

        ROI_use = False

        if random_i == 0:
            save_folder = os.path.join(save_folder_main, 'static')
        else:
            save_folder = os.path.join(save_folder_main, 'sim_'+str(random_i))
        ff.make_folder([save_folder])

        if os.path.isfile(os.path.join(save_folder, 'img.nii.gz')) == 1:
            print('already exist')
            continue

        if random_i == 0:
            nb.save(nb.Nifti1Image(img, np.eye(4)), os.path.join(save_folder, 'img.nii.gz'))
            continue

        # Randomly pick one value from PSFsize_list
        PSFsize = random.choice(PSFsize_list)
        print("Randomly picked PSFsize:", PSFsize)

        # Randomly pick one value from limited_displacement_list
        limited_displacement = random.choice(limited_displacement_list)
        print("Randomly picked limited_displacement:", limited_displacement)
        limited_displacement = limited_displacement / pixel_spacing[0]

        MaxTotalLength = np.random.uniform(MaxTotalLength_range[0], MaxTotalLength_range[1]) / pixel_spacing[0]
        print("Randomly picked MaxTotalLength:", MaxTotalLength)

        exposure_time = [1]

        # create motion trajectory
        TrajCurve = ff.create_motion_trajectory(PSFsize, anxiety, numT, MaxTotalLength,limited_displacement, plot_traj=False)

        # create PSF for each exposure time
        PSFS = ff.create_PSF(TrajCurve, exposure_time, PSFsize, plot_PSF=False)

        # ROI
        if np.random.uniform(0,1) < ROI_frequency:
            print('use ROI')
            ROI = ff.create_random_ROI(img, radius_range = [img.shape[0]//5, img.shape[0]//3],center_of_mass = center_of_mass, plot_ROI=False)
            ROI_copy = np.copy(ROI)
            smooth_ROI = cv2.GaussianBlur(ROI_copy.astype(float), (51,51), 10)
            ROI_use = True
        else:
            print('no ROI')
            ROI = np.ones_like(img)
            smooth_ROI = np.ones_like(img)
            ROI_use = False
        

        # create final blurred image
        final_img = ff.create_motion_blur_img(img, PSFS, add_noise=False, sigma_gauss=0.05)[:,:,0]
        # scale the final image so that keep the intensity range same
        final_img = (final_img - np.min(final_img)) / (np.max(final_img) - np.min(final_img)) * (np.max(img) - np.min(img)) + np.min(img)

        # some chances that only part of the image is blurred
        if ROI_use:
            # use smooth ROI
            final_img_roi = final_img * smooth_ROI + img * (1 - smooth_ROI)
        else:
            final_img_roi = final_img

        # make sure the background is always 0
        final_img[img<=0] = img[img<=0]; final_img = np.round(final_img)
        final_img_roi[img<=0] = img[img<=0]; final_img_roi = np.round(final_img_roi)

        # save the image and ROI
        nb.save(nb.Nifti1Image(final_img_roi, np.eye(4)), os.path.join(save_folder, 'img.nii.gz'))
        nb.save(nb.Nifti1Image(final_img, np.eye(4)), os.path.join(save_folder, 'img_entireblur_as_reference.nii.gz'))
        ROI = ROI.astype(np.uint8)
        nb.save(nb.Nifti1Image(ROI, np.eye(4)), os.path.join(save_folder, 'ROI.nii.gz'))
        nb.save(nb.Nifti1Image(smooth_ROI, np.eye(4)), os.path.join(save_folder, 'smooth_ROI.nii.gz'))

        # save info
        result.append([patient['Image'], random_i, ROI_use, PSFsize[0], limited_displacement * pixel_spacing[0], MaxTotalLength * pixel_spacing[0], patient['Blur'], patient['Agree'], patient['PoorQuality'],patient['AdjuDisagree'], patient['Folder'], patient['MRN'], patient['SegFilename'], patient['Dataset']])

        df = pd.DataFrame(result, columns = ['Image', 'simulation', 'use_ROI?', 'PSFsize', 'limited_displacement(mm)', 'MaxTotalLength(mm)', 'Blur_in_original_image', 'Agree', 'PoorQuality', 'AdjuDisagree', 'Folder', 'MRN', 'SegFilename', 'Dataset'])
        df.to_excel(os.path.join(os.path.dirname(save_folder_main), 'simulation_info.xlsx'), index = False)


# for blur: jus read dicom and save as nii
# result = []
# for index in range(0, sheet.shape[0]):
#     patient = sheet.iloc[index]
#     patient_image_name = patient['Image']
#     print('this file name is: ', patient_image_name)
#     patient_file = os.path.join('/mnt/BPM_NAS', patient['Folder'], 'data_lut', patient_image_name)

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

#     save_folder = os.path.join('/mnt/BPM_NAS/real_blurs', patient_image_name, 'original_img')
#     ff.make_folder([os.path.dirname(save_folder), save_folder])

#     if os.path.isfile(os.path.join(save_folder, 'img.nii.gz')) == 0:
#         nb.save(nb.Nifti1Image(img, np.eye(4)), os.path.join(save_folder, 'img.nii.gz'))

#     result.append([patient['Image'],  patient['Blur'], patient['Agree'], patient['PoorQuality'],patient['AdjuDisagree'], patient['Folder'], patient['MRN'], patient['SegFilename'], patient['Dataset']])

#     df = pd.DataFrame(result, columns = ['Image', 'Blur_in_original_image', 'Agree', 'PoorQuality', 'AdjuDisagree', 'Folder', 'MRN', 'SegFilename', 'Dataset'])
#     df.to_excel(os.path.join(os.path.dirname(os.path.dirname(save_folder)), 'blur_info.xlsx'), index = False)


# To correct the simulation: set the ROI in the region of breast,smooth the ROI boundary, round the value, make sure the background is 0
# for index in range(0, 1):#sheet.shape[0]):
#     patient = sheet.iloc[index]
#     patient_image_name = patient['Image']
#     print(patient_image_name)

#     original_img = nb.load(os.path.join('/mnt/BPM_NAS/simulations', patient_image_name, 'static', 'img.nii.gz')).get_fdata()
    
#     for random_i in range(1,4):
#         folder = os.path.join('/mnt/BPM_NAS/simulations', patient_image_name, 'sim_'+str(random_i))
#         simulated_img = nb.load(os.path.join(folder, 'img.nii.gz'))
#         affine = simulated_img.affine
#         simulated_img = simulated_img.get_fdata()
#         ROI = nb.load(os.path.join(folder, 'ROI.nii.gz')).get_fdata()
#         print(np.unique(ROI))

#         # Check if every value in ROI is 1
#         if np.all(ROI == 1):
#             print("Every value in ROI is 1")
#         else:
#             print("Not every value in ROI is 1, then smooth this ROI")
#             ROI_copy = np.copy(ROI)
#             smooth_ROI = cv2.GaussianBlur(ROI_copy.astype(float), (51,51), 10)
#             final_img_roi_smooth = simulated_img * smooth_ROI + img * (1 - smooth_ROI)
        
#         simulated_img[original_img<=0] = original_img[original_img<=0]
#         simulated_img = np.round(simulated_img)
#         nb.save(nb.Nifti1Image(simulated_img, affine), os.path.join(folder, 'img_corrected.nii.gz'))
            
            



       
    

    




      

        
