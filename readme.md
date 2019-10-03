# Introduction 

This framework designed for medical image segmentation. It uses the ushaped deep CNN architecture (VNet or UNet) extended by attention gates. For more information about the method, please read the following paper:

`The paper describing the method is still under the review.`

Please also cite this paper if you are using this code for your research!

The attention mechanism helps the network to focus on the desired area of CT abdomen scan and therefore improves the accuracy. The attention maps from two network levels (image resolutions) for three different abdomen CT images datasets are in Fig 1 below.

![att_map](https://user-images.githubusercontent.com/32820891/65874932-cd3d9a00-e386-11e9-8fa4-b59e419285c3.png)

This repository is still work in progress. Things may break. If that is the case, please let us know.
The code was wastly inspired by the nnU-Net framework authored by Fabian Isensee, so in case of troubles you may want to study this repository and its describtion, too: https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunet.

# Installation 
The instalation is only tested on Linux (Ubuntu). It may work on other operating systems as well but we do not guarantee it will.

Installation instructions
1) Install PyTorch (https://pytorch.org/get-started/locally/)
2) Clone this repository `git clone https://github.com/tureckova/Abdomen-CT-Image-Segmentation`
3) Go into the repository (`cd Abdomen-CT-Image-Segmentation` on linux)
4) Install with `pip install -r requirements.txt` followed by `pip install -e .`

# Getting Started 
All the commands in this section assume that you are in a terminal and your working directory is the repository folder 
(the one that has all the subfolders like `dataset_conversion`, `evaluation`, ...)

## Set paths 
Framwork needs to know where you will store raw data, want it to store preprocessed data and trained models. Have a 
look at the file `paths.py` and adapt it to your system.

## Preparing Datasets 
The preprocessing pipeline was adapted from nnU-Net. Please refer to the readme.md in the 
 `dataset_conversion` subfolder for detailed information. Examples are also provided there. You will need to 
 convert your dataset into this format before you can continue.
 
Place your dataset either in the `raw_dataset_dir` or `splitted_4d_output_dir`, as specified in `paths.py` (depending on how you prepared it, again 
see the readme in `dataset_conversion`). Give 
it a name like: `TaskXX_MY_DATASET` (where XX is some number) to be consistent with the naming scheme of the Medical 
Segmentation Decathlon.

## Experiment Planning and Preprocessing 
Framework can now analyze your dataset and determine how to train its models. To run experiment planning and preprocessing for your dataset, execute the following command:

`python3 experiment_planning/plan_and_preprocess_task.py -t Task07_Pancreas -p 8`

here `TaskXX_MY_DATASET` specifies the task (your dataset) and `-p` determines how many processes will be used for 
datatset analysis and preprocessing. Generally you want this number to be as high as you have CPU cores, unless you 
run into memory problems (beware of datasets such as LiTS!)

Running this command will to several things:
1) If you stored your data as 4D nifti the data will be split into a sequence of 3d niftis. Back when I started 
SimpleITK did not support 4D niftis. This was simply done out of necessity.
2) Images are cropped to the nonzero region. This does nothing for most datasets. Most brain datasets however are brain 
extracted, meaning that the brain is surrounded by zeros. There is no need to push all these zeros through the GPUs so 
we simply save a little bit of time doing this. Cropped data is stored in the `cropped_output_dir` (`paths.py`).
3) The data is analyzed and information about spacing, intensity distributions and shapes are determined
4) nnU-Net configures the architectures based on that information. All U-Nets are configured to optimally use 
**12GB Nvidia TitanX** GPUs.
5) The preprocessing is run and it saves the preprocessed data and plans files in `preprocessing_output_dir`. **You could accomodate the plans files (for example to fit in smaller GPU), the skript rewrite_pickle_plan.py may help you.**

I strongly recommend you set `preprocessing_output_dir` on a SSD. HDDs are typically too slow for data loading. Plan files as we use it is the paper here in the folder plans_files.

## Training Models
In our paper we compared two variants of 3D CNN - VNet or UNet and two resolution variants - full-resolution and low-resolution. The default setting is to train each of these models in a five-fold cross-validation.

Trained models are stored in `network_training_output_dir` (specified in `paths.py`).

For `FOLD` in [0, 4], run:

`python run/run_training.py [3d_fullres/3d_lowres] nnUNetTrainer TaskXX_MY_DATASET FOLD --ndet --vnet=[0/1]`

you need to choose only one of variants in [] and remove the brackets.

You can continue the already started training of the model by adding --continue_training to the command. The model will recover from the newest checkpoint.

## Evaluation of models
The model is evaluated automaticaly in the end of the training, but the model could be only evaluated by the same command as training only with added `-val` in the end of it.

## Inference 
You can use trained models to predict test data. In order to be able to do so the test data must be provided in the 
same format as the training data. 

To run inference use the following script:

`python inference/predict_simple.py -i INPUT_FOLDER -o OUTPUT_FOLDER -t TaskXX_MY_DATASET -tr nnUNetTrainer -m [3d_fullres/3d_lowres] --vnet=[0/1]`

If you wish to ensemble different inference cases, run all inference commands with the `-z` argument. This will tell the framework to save the softmax probabilities as well. They are needed for ensembling.

### Ensembling predicted results
You can then ensemble the predictions of two output folders (there must be saved the softmax probabilities, see above) with the following command:

`python inference/ensemble_predictions.py -f FOLDER1 FODLER2 ... -o OUTPUT_FOLDER`

This will ensemble the predictions located in `FODLER1, FOLDER2, ...` and write them into `OUTPUT_FOLDER`

## Tips and Tricks
 
#### Manual Splitting of Data
The cross-validation in nnU-Net splits on a per-case basis. This may sometimes not be desired, for example because 
several training cases may be the same patient (different time steps or annotators). If this is the case, then you need to
manually create a split file. To do this, first let nnU-Net create the default split file. Run one of the network 
trainings (any of them works fine for this) and abort after the first epoch. nnU-Net will have created a split file automatically:
`preprocessing_output_dir/TaskXX_MY_DATASET/splits_final.pkl`. This file contains a list (length 5, one entry per fold). 
Each entry in the list is a dictionary with keys 'train' and 'val' pointing to the patientIDs assigned to the sets. 
To use your own splits in nnU-Net, you need to edit these entries to what you want them to be and save it back to the 
splits_final.pkl file. Use load_pickle and save_pickle from batchgenerators.utilities.file_and_folder_operations for convenience.

#### Sharing Models
You can share trained models by simply sending the corresponding output folder from `network_training_output_dir` to 
whoever you want share them with. The recipient can then use nnU-Net for inference with this model.
