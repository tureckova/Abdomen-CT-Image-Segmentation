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

import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.run.default_configuration import get_default_configuration
from nnunet.paths import default_plans_identifier
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.training.network_training.nnUNetTrainerCascadeFullRes import nnUNetTrainerCascadeFullRes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("network_trainer")
    parser.add_argument("task")
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier", default=default_plans_identifier, required=False)
    parser.add_argument("-u", "--unpack_data", help="Leave it as 1, development only", required=False, default=1,
                        type=int)
    parser.add_argument("--ndet", help="Per default training is deterministic, "
                                                   "nondeterministic allows cudnn.benchmark which will can give up to "
                                                   "20%% performance. Set this to do nondeterministic training",
                        required=False, default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the vlaidation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true", help="not used here, just for fun")
    parser.add_argument("--valbest", required=False, default=False, action="store_true", help="hands off. This is not intended to be used")
    parser.add_argument("--fp16", required=False, default=False, action="store_true", help="enable fp16 training. Makes sense for 2d only! (and only on supported hardware!)")
    parser.add_argument("--use_label", required=False, default=None, help="if None uses all default labels,"
                                                                          "if organ uses only label marked as 1,"
                                                                          "if tumor uses only label marked as 2,"
                                                                          "if both unites label 1 and 2 into one label")
    parser.add_argument("--vnet", required=False, default=1, type=int,
                        help="If 1 uses convolutional pooling and upsampling, if 0 uses maxpooling and upsampling")
    parser.add_argument("--init_model", required=False, default=None, help="Path to trained model which will be used to initialize weights of the network before training")
    parser.add_argument("--freeze_ag", required=False, default=False, help="If True the weights in Attention gates in model will be freezed")
    parser.add_argument("--save_attention_map", required=False, default=False, help="if True, saves attention maps for each image into separate folder")
    parser.add_argument("--save_ds_map", required=False, default=False, help="if True, saves deep supervision maps for each image into separate folder")

    args = parser.parse_args()

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only
    plans_identifier = args.p
    find_lr = args.find_lr
    unpack = args.unpack_data
    deterministic = not args.ndet
    valbest = args.valbest
    fp16 = args.fp16
    use_label = args.use_label
    save_attention_map = args.save_attention_map
    save_ds_map = args.save_ds_map
    vnet = args.vnet
    print('vnet: ', vnet)

    if unpack == 0:
        unpack = False
    elif unpack == 1:
        unpack = True
    else:
        raise ValueError("Unexpected value for -u/--unpack_data: %s. Use 1 or 0." % str(unpack))

    if vnet == 0:
        vnet = False
    elif vnet == 1:
        vnet == True
    else:
        raise ValueError("Unexpected value for --vnet %s. Use 1 or 0." % str(vnet))

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier)

    if trainer_class is None:
        raise RuntimeError("Could not find trainer class in nnunet.training.network_training")

    if network == "3d_cascade_fullres":
        assert issubclass(trainer_class, nnUNetTrainerCascadeFullRes), "If running 3d_cascade_fullres then your " \
                                                                       "trainer class must be derived from " \
                                                                       "nnUNetTrainerCascadeFullRes"
    else:
        assert issubclass(trainer_class, nnUNetTrainer), "network_trainer was found but is not derived from nnUNetTrainer"

    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=unpack, deterministic=deterministic,
                            fp16=fp16, use_label=use_label, vnet=vnet)

    trainer.initialize(not validation_only)

    if find_lr:
        trainer.find_lr()
    else:
        if not validation_only:
            if args.continue_training:
                trainer.load_latest_checkpoint()
            else:
                if args.init_model is not None:
                    trainer.init_model(args.init_model)
                if args.freeze_ag:
                    trainer.freeze_ag()
                    trainer.initial_lr = 3e-5
            trainer.run_training()
        elif not valbest:
            trainer.load_latest_checkpoint(train=False)

        if valbest:
            trainer.load_best_checkpoint(train=False)
            val_folder = "validation_best_epoch"
        else:
            val_folder = "validation"

        # predict validation
        if save_attention_map:
            trainer.save_attention(override=True)
        if save_ds_map:
            trainer.save_ds(override=True)
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder, override=False)

        # if network == '3d_lowres':
        #     trainer.load_best_checkpoint(False)
        #     print("predicting segmentations for the next stage of the cascade")
        #     predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))
