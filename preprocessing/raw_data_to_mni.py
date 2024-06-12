#########################################################
# Transform coregistered scans from the subject space to
# the MNI305 space
#########################################################

import os
os.environ["SUBJECTS_DIR"] = "/mnt/HD1/milad/multimodal_decoding/freesurfer/subjects"

from glob import glob
from os.path import join as opj

subject  = 'sub-03'
nipype_subject = f'_subject_id_{subject}'

vol_dir  = f'/mnt/HD1/milad/multimodal_decoding/preprocessed_data/preprocess_workflow/{nipype_subject}'
reg_file = f'/mnt/HD1/milad/multimodal_decoding/freesurfer/regfiles/{subject}/spm2fs.change-name.lta'
out_dir  = f'/mnt/HD1/milad/multimodal_decoding/preprocessed_data/mni305/{subject}'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

all_vol_files = list(sorted(glob(opj(vol_dir, '_session_*', 'coregister', f'rara{subject}_*_bold.nii'))))
sess_visit = set()
for vol_file in all_vol_files:
    idx = vol_file.find('ses-')
    sess_id = vol_file[idx:idx+6]
    out_sess_dir = opj(out_dir, sess_id)
    if sess_id not in sess_visit:
        sess_visit.add(sess_id)
        if not os.path.exists(out_sess_dir):
            os.makedirs(out_sess_dir)
    file_name = os.path.basename(vol_file)
    out_vol = opj(out_sess_dir, file_name)

    conv_cmd = f'mri_vol2vol --mov "{vol_file}" --reg "{reg_file}" --o "{out_vol}" --tal --talres 2'
    os.system(conv_cmd)
pass