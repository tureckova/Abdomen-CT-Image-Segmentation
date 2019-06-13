#    Copyright 2019 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import numpy as np


def convert_to_submission(source_dir, target_dir):
    niftis = subfiles(source_dir, join=False, suffix=".nii.gz")
    patientids = np.unique([i[:10] for i in niftis])
    maybe_mkdir_p(target_dir)
    for p in patientids:
        files_of_that_patient = subfiles(source_dir, prefix=p, suffix=".nii.gz", join=False)
        assert len(files_of_that_patient)
        files_of_that_patient.sort()
        # first is ED, second is ES
        shutil.copy(join(source_dir, files_of_that_patient[0]), join(target_dir, p + "_ED.nii.gz"))
        shutil.copy(join(source_dir, files_of_that_patient[1]), join(target_dir, p + "_ES.nii.gz"))


if __name__ == "__main__":
    folder = "/home/tureckova/Pictures/kits19/data"
    folder_test = "/media/fabian/My Book/datasets/ACDC/testing/testing"
    out_folder = "/home/tureckova/Pictures/nnUNet/nnUNet_base/nnUNet_raw_splitted/KiTS2019"

    maybe_mkdir_p(join(out_folder, "imagesTr"))
    maybe_mkdir_p(join(out_folder, "imagesTs"))
    maybe_mkdir_p(join(out_folder, "labelsTr"))

    # train
    all_train_files = []
    patient_dirs_train = subfolders(folder, prefix="case")
    for current_dir in patient_dirs_train:
        data_file_train = current_dir + "/imaging.nii.gz"
        seg_file = current_dir + "/segmentation.nii.gz"
        patient_identifier = "KiTS2019_" + current_dir[-3:]
        all_train_files.append(patient_identifier + "_0000.nii.gz")
        shutil.copy(data_file_train, join(out_folder, "imagesTr", patient_identifier + "_0000.nii.gz"))
        shutil.copy(seg_file, join(out_folder, "labelsTr", patient_identifier + ".nii.gz"))

    # test
    # all_test_files = []
    # patient_dirs_test = subfolders(folder_test, prefix="patient")
    # for p in patient_dirs_test:
    #     current_dir = p
    #     data_files_test = [i for i in subfiles(current_dir, suffix=".nii.gz") if i.find("_gt") == -1 and i.find("_4d") == -1]
    #     for d in data_files_test:
    #         patient_identifier = d.split("/")[-1][:-7]
    #         all_test_files.append(patient_identifier + "_0000.nii.gz")
    #         shutil.copy(d, join(out_folder, "imagesTs", patient_identifier + "_0000.nii.gz"))


    json_dict = OrderedDict()
    json_dict['name'] = "KiTS2019"
    json_dict['description'] = "the 2019 Kidney and Kidney Tumor Segmentation Challenge"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "***"
    json_dict['licence'] = "MIT License"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Liver",
        "2": "Tumor"
    }
    json_dict['numTraining'] = len(all_train_files)
    json_dict['numTest'] = 0 #len(all_test_files)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1][:-12], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1][:-12]} for i in
                             all_train_files]
    json_dict['test'] = []#["./imagesTs/%s.nii.gz" % i.split("/")[-1][:-12] for i in all_test_files]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))
