import SAM_CMR_seg.functions_collection as ff

import os
import numpy as np
import nibabel as nb
import pandas as pd
import shutil
import SimpleITK as sitk


# delete files
patient_list = ff.find_all_target_files(['*/sim_4', '*/sim_5'],'/mnt/BPM_NAS/simulations')
for patient in patient_list:
    shutil.rmtree(patient)
