from batchgenerators.utilities.file_and_folder_operations import load_pickle
from batchgenerators.utilities.file_and_folder_operations import save_pickle

import numpy as np

from pathlib import Path

plans_file = Path.joinpath(Path.home(),"Pictures/nnUNet/nnUNet_preprocessed/KiTS2019/nnUNetPlans_plans_3D.pkl")
new_plans_file = Path.joinpath(Path.home(),"Pictures/nnUNet/nnUNet_preprocessed/KiTS2019/nnUNetPlans_plans_3D.pkl")

plans = load_pickle(plans_file)

plans['plans_per_stage'][0]['patch_size'] = np.array([128, 128,  80])
plans['plans_per_stage'][0]['patch_size_org'] = np.array([160, 128,  80])
#plans['num_classes'] = 1
#plans['all_classes'] = np.array([2])

save_pickle(plans, new_plans_file)
