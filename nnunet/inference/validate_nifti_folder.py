from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.evaluation.evaluator import aggregate_scores


def validate(folder, gt_folder):

    patient_ids = subfiles(folder, suffix=".nii.gz", join=False)

    pred_gt_tuples = []
    for p in patient_ids:
        file = join(folder, p)
        gt_file = join(gt_folder, p)
        pred_gt_tuples.append([file, gt_file])

    task = folder.split("/")[-4]
    job_name = 'esembly fullres and lowres'
    num_classes = 3
    _ = aggregate_scores(pred_gt_tuples, labels=list(range(num_classes)),
                         use_label=None,
                         json_output_file=join(folder, "summary.json"),
                         json_name=job_name + folder.split("/")[-2],
                         json_author="Bety",
                         json_task=task, num_threads=3)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Computes metrics scores for validation folder containing nifti files")
    parser.add_argument('-f', '--folder', type=str, help="Folder with nifti files to evaluate", required=True)
    parser.add_argument('-gtf', '--gtfolder', type=str, help="Folder with GT nifti files.", required=False)
    args = parser.parse_args()

    folder = args.folder
    gt_folder = args.gtfolder

    validate(folder, gt_folder)