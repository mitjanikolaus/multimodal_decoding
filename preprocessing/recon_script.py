###################################
# Freesurfer Recon-all for subjects
# Runs recon-all for all subjects in parallel
###################################
import argparse
import os
import subprocess

from utils import FREESURFER_BASE_DIR, SUBJECTS, FMRI_RAW_DATA_DIR


def run(args):
    # os.environ["SUBJECTS_DIR"] = f"{FREESURFER_BASE_DIR}/subjects"
    os.environ["SUBJECTS_DIR"] = f"{FREESURFER_BASE_DIR}/subjects_downsampled"

    processes = []
    std_out_files = []
    std_err_files = []
    for subject in args.subjects:
        # anat_path = os.path.join(FMRI_RAW_DATA_DIR, 'corrected_anat', subject, f'{subject}_ses-01_run-01_T1W.nii')
        anat_path = os.path.join(FMRI_RAW_DATA_DIR, 'corrected_anat', subject, f'{subject}_ses-01_run-01_T1W_downsampled.nii')
        assert os.path.isfile(anat_path)
        outf = open(f'recon_out_{subject}.txt', 'w')
        errf = open(f'recon_err_{subject}.txt', 'w')
        std_out_files.append(outf)
        std_err_files.append(errf)
        processes.append(
            subprocess.Popen(["recon-all", "-s", subject, "-i", anat_path, "-all"], stdout=outf, stderr=errf)
        )

    print('waiting for the processes to be done ...')
    for p in processes:
        p.communicate()

    for f in std_out_files:
        f.close()

    for f in std_err_files:
        f.close()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
