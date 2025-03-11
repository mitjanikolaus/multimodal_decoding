import argparse
import os
import numpy as np
# from nipype.interfaces.fsl import ApplyMask,
from nipype.interfaces.spm import SliceTiming, Realign, Coregister, NewSegment, Normalize12, Threshold
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node
from nipype import MapNode
from nipype.algorithms.misc import Gunzip

import nipype.interfaces.matlab as mlab

from utils import FMRI_RAW_DATA_DIR, FMRI_PREPROCESSED_DATA_DIR, SUBJECTS

SPM_PATH = os.path.expanduser('~/apps/spm12')
mlab.MatlabCommand.set_default_paths(SPM_PATH)


def print_session_names(sessions):
    for subject in sessions:
        print(subject)
        print(sessions[subject])
        print()


def print_run_names(runs):
    for subject, session in runs:
        print(subject, session)
        for run in runs[subject, session]:
            print(run)
        print()


def run(args):
    subjects = args.subjects
    print(subjects)
    print()

    # list subject sessions
    data_root = os.path.join(args.raw_data_dir, 'bids')
    anat_root = os.path.join(args.raw_data_dir, 'corrected_anat')
    sessions = dict()
    for subj in subjects:
        folders = os.listdir(os.path.join(data_root, subj))
        sessions[subj] = sorted(folders)
    print_session_names(sessions)

    # list functional runs
    runs = dict()
    for subj in subjects:
        for ses in sessions[subj]:
            rns = []
            files = os.listdir(os.path.join(data_root, subj, ses, 'func'))
            for file in files:
                if file.endswith("bold.nii.gz"):
                    rns.append(os.path.join(data_root, subj, ses, 'func', file[:-12]))
            runs[(subj, ses)] = sorted(rns)
    print_run_names(runs)

    # fMRI setting
    TR = 2
    number_of_slices = 46
    ref_slice_index = 22
    multiband_factor = 2
    interval = TR / (number_of_slices / multiband_factor)

    slice2time = [0] * number_of_slices
    time = interval * 1000
    for f, temp in enumerate([[0, 23], [1, 24]]):
        for i in range(12 - f):
            slice2time[temp[0] + i * 2] = min(time, TR * 1000)
            slice2time[temp[1] + i * 2] = min(time, TR * 1000)
            time += interval * 1000

    for idx, t in enumerate(slice2time):
        print(f"{idx:02d} {t:10.4f}")

    toprint = np.array(slice2time).argsort()
    for i in range(0, 46, 2):
        print(f"{toprint[i] + 1:02d} {toprint[i + 1] + 1:02d} {slice2time[toprint[i]]:10.4f}")

    ##############
    # Nipype Nodes
    ##############
    # Gunzip to unpack tar.gz
    gunzip_func_node = MapNode(Gunzip(), iterfield=['in_file'], name='gunzip_func')

    # Slice timing correction
    stc_node = Node(
        SliceTiming(
            num_slices=number_of_slices,
            time_repetition=TR,
            time_acquisition=TR - (TR / (number_of_slices / multiband_factor)),
            slice_order=slice2time,
            ref_slice=slice2time[ref_slice_index],
        ),
        name='stc'
    )

    # Realignment
    realign_node = Node(Realign(register_to_mean=True), name='realign')

    # Coregistration (coregistration of functional scans to anatomical scan)
    coregister_node = Node(Coregister(jobtype='estimate'), name='coregister')

    # Normalization (transformation to MNI space)
    template = os.path.join(SPM_PATH, 'tpm/TPM.nii')  # template in form of a tissue probability map to normalize to
    normalize = Node(Normalize12(tpm=template, write_voxel_sizes=[2, 2, 2]), name="normalize")
    # normalize_anat = Node(Normalize12(out_prefix='n', tpm=template, write_voxel_sizes=[2, 2, 2]), name="normalize_anat")

    # template = os.path.join(SPM_PATH, 'canonical/avg305T1.nii')
    # normalize_node = Node(DARTELNorm2MNI(modulate=True, template_file=template, voxel_size=[2, 2, 2]), name='normalize')
    # normalize_node.iterables = ('fwhm', [4])

    # template = os.path.join(SPM_PATH, 'canonical/avg305T1.nii')
    # normalize_func = Node(DARTELNorm2MNI(modulate=True, template_file=template, voxel_size=(2.0, 2.0, 2.0)),
    #                       name='normalize_func')
    # fwhmlist = [4]
    # normalize_and_smooth_func.iterables = ('fwhm', fwhmlist)

    # normalize_anat = Node(DARTELNorm2MNI(modulate=True, template_file=template, voxel_size=(2.0, 2.0, 2.0)),
    #                         name='normalize_struct')
    # normalize_struct.inputs.fwhm = 2

    tmp_img = os.path.join(SPM_PATH, "tpm/TPM.nii")
    tissue1 = ((tmp_img, 1), 2, (True, True), (False, False))
    tissue2 = ((tmp_img, 2), 2, (True, True), (False, False))
    tissue3 = ((tmp_img, 3), 2, (True, False), (False, False))
    tissue4 = ((tmp_img, 4), 2, (False, False), (False, False))
    tissue5 = ((tmp_img, 5), 2, (False, False), (False, False))
    tissues = [tissue1, tissue2, tissue3, tissue4, tissue5]
    segment_node = Node(NewSegment(tissues=tissues), name='segment')

    # Threshold - Threshold GM probability image
    # mask_GM = Node(Threshold(thresh=0.0,
    #                          args='-bin -dilF',
    #                          output_type='NIFTI'),
    #                name="mask_GM")
    #
    # mask_func = MapNode(ApplyMask(output_type='NIFTI'),
    #                     name="mask_func",
    #                     iterfield=["in_file"])

    # Info source (to provide input information to the pipeline)
    # to iterate over subjects
    infosrc_subjects = Node(IdentityInterface(fields=['subject_id']), name="infosrc_subjects")
    infosrc_subjects.iterables = [('subject_id', subjects)]

    # to iterate over sessions of each subject
    infosrc_sessions = Node(IdentityInterface(fields=['session_id']), name="infosrc_sessions")
    infosrc_sessions.itersource = ("infosrc_subjects", 'subject_id')
    infosrc_sessions.iterables = [('session_id', sessions)]

    # File selector (to list files for the pipeline based on the info sources)
    anat_file = os.path.join('{subject_id}', '{subject_id}_ses-01_run-01_T1W.nii')
    func_file = os.path.join('{subject_id}', '{session_id}', 'func', '*bold.nii.gz')

    selectfiles_anat = Node(
        SelectFiles({'anat': anat_file}, base_directory=anat_root), name="selectfiles_anat"
    )

    selectfiles_sessions = Node(
        SelectFiles({'func': func_file}, base_directory=data_root), name="selectfiles_sessions"
    )

    # Working directory
    workflow_dir = args.out_data_dir
    os.makedirs(workflow_dir, exist_ok=True)

    # Datasink - creates an extra output folder for storing the desired files
    datasink_node = Node(DataSink(base_directory=workflow_dir, container='datasink'), name="datasink")

    # Remove nipype's prefix for the files and folders in the datasink
    substitutions = [('_subject_id_', ''), ('_session_id_', '')]
    datasink_node.inputs.substitutions = substitutions

    #################################################
    # Workflow (building the pipeline with the nodes)
    #################################################
    # create the workflow
    preproc = Workflow(name='preprocess_workflow')
    preproc.base_dir = workflow_dir
    # preproc.config['execution'] = {} # you can add configurations here

    # connect info source to file selectors
    preproc.connect([(infosrc_subjects, selectfiles_anat, [('subject_id', 'subject_id')])])
    preproc.connect([(infosrc_subjects, infosrc_sessions, [('subject_id', 'subject_id')])])
    preproc.connect([(infosrc_sessions, selectfiles_sessions, [('session_id', 'session_id')])])
    preproc.connect([(infosrc_subjects, selectfiles_sessions, [('subject_id', 'subject_id')])])

    # connect file selectors to gunzip
    preproc.connect([(selectfiles_sessions, gunzip_func_node, [('func', 'in_file')])])

    # connect gunzip to STC
    preproc.connect([(gunzip_func_node, stc_node, [('out_file', 'in_files')])])

    # connect STC to realign
    preproc.connect([(stc_node, realign_node, [('timecorrected_files', 'in_files')])])

    # connect realign to coregister
    preproc.connect([(realign_node, coregister_node, [('mean_image', 'source')])])
    preproc.connect([(realign_node, coregister_node, [('realigned_files', 'apply_to_files')])])
    preproc.connect([(selectfiles_anat, coregister_node, [('anat', 'target')])])

    # connect coregister to normalize
    preproc.connect([(selectfiles_anat, normalize, [('anat', 'image_to_align')])])
    preproc.connect([(coregister_node, normalize, [('coregistered_files', 'apply_to_files')])])

    # connect segment
    preproc.connect([(normalize, segment_node, [('normalized_image', 'channel_files')])])

    # Select GM segmentation file from segmentation output
    def get_gm(files):
        return files[0][0]

    # connect threshold
    # preproc.connect([(segment_node, mask_GM, [(('native_class_images', get_gm), 'in_file')])])
    #
    # preproc.connect([(normalize, mask_func, [('normalized_files', 'in_file')]),
    #                  (mask_GM, mask_func, [('out_file', 'mask_file')])
    #                  ])

    # keeping realignment params
    preproc.connect([(realign_node, datasink_node, [('realignment_parameters', 'realignment.@par')])])

    # draw graph of the pipeline
    preproc.write_graph(graph2use='flat', format='png', simple_form=True)

    # run thef pipeline
    preproc.run('MultiProc', plugin_args={'n_procs': 15})


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--raw-data-dir", type=str, default=FMRI_RAW_DATA_DIR)
    parser.add_argument("--out-data-dir", type=str, default=FMRI_PREPROCESSED_DATA_DIR)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
