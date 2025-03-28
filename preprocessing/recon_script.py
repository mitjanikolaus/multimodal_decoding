import argparse
import os
import subprocess

from utils import FREESURFER_SUBJECTS_DIR, FMRI_RAW_DATA_DIR, SUBJECTS


def run(args):
    os.environ["SUBJECTS_DIR"] = FREESURFER_SUBJECTS_DIR
    os.makedirs(FREESURFER_SUBJECTS_DIR, exist_ok=True)

    processes = []
    std_out_files = []
    std_err_files = []
    assert os.path.isfile(args.anat_scan_path)
    outf = open(f'recon_out_{args.subject}.txt', 'w')
    errf = open(f'recon_err_{args.subject}.txt', 'w')
    std_out_files.append(outf)
    std_err_files.append(errf)
    processes.append(
        subprocess.Popen(["recon-all", "-s", args.subject, "-i", args.anat_path, "-all"], stdout=outf, stderr=errf)
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

    default_path = os.path.join(FMRI_RAW_DATA_DIR, 'corrected_anat', SUBJECTS[0],
                                f'{SUBJECTS[0]}_ses-01_run-01_T1W.nii')
    # _downsampled_2mm
    parser.add_argument("--anat-scan-path", type=str, default=default_path)

    parser.add_argument("--subject", type=str, default=SUBJECTS[0])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
