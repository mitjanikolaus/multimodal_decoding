import argparse
import os
from nipype.interfaces.spm import SliceTiming, Realign, Coregister, NewSegment
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.pipeline.engine import Workflow, Node
from nipype import MapNode
from nipype.algorithms.misc import Gunzip

import nipype.interfaces.matlab as mlab

from utils import FMRI_PREPROCESSED_DATA_DIR, SUBJECTS, FMRI_BIDS_DATA_DIR, FMRI_DOWNSAMPLED_ANAT_DATA_DIR

SPM_PATH = os.path.expanduser('~/apps/spm12')
mlab.MatlabCommand.set_default_paths(SPM_PATH)


DEFAULT_ANAT_SCAN_SUFFIX = "_downsampled_2mm"


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
    sessions = dict()
    for subj in subjects:
        if args.sessions is not None:
            sessions[subj] = args.sessions
        else:
            folders = os.listdir(os.path.join(args.bids_data_dir, subj))
            sessions[subj] = sorted(folders)
    print_session_names(sessions)

    # list functional runs
    runs = dict()
    for subj in subjects:
        for ses in sessions[subj]:
            rns = []
            files = os.listdir(os.path.join(args.bids_data_dir, subj, ses, 'func'))
            for file in files:
                if file.endswith("bold.nii.gz"):
                    rns.append(os.path.join(args.bids_data_dir, subj, ses, 'func', file[:-12]))
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

    print("Slice Timing: ")
    for idx, t in enumerate(slice2time):
        # print(f"{idx:02d} {t:10.4f}")
        print(f"{t/1000:.4f}", end=', ')

    # toprint = np.array(slice2time).argsort()
    # for i in range(0, 46, 2):
    #     print(f"{toprint[i] + 1:02d} {toprint[i + 1] + 1:02d} {slice2time[toprint[i]]:10.4f}")

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
    coregister_node = Node(Coregister(jobtype='estwrite'), name='coregister')

    tpm_img = os.path.join(SPM_PATH, "tpm/TPM.nii")
    tissue1 = ((tpm_img, 1), 2, (True, False), (False, False))
    tissue2 = ((tpm_img, 2), 2, (True, False), (False, False))
    tissue3 = ((tpm_img, 3), 2, (True, False), (False, False))
    tissue4 = ((tpm_img, 4), 2, (False, False), (False, False))
    tissue5 = ((tpm_img, 5), 2, (False, False), (False, False))
    tissue6 = ((tpm_img, 6), 2, (False, False), (False, False))
    tissues = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]
    segment_node = Node(NewSegment(tissues=tissues), name='segment')

    # Info source (to provide input information to the pipeline)
    # to iterate over subjects
    infosrc_subjects = Node(IdentityInterface(fields=['subject_id']), name="infosrc_subjects")
    infosrc_subjects.iterables = [('subject_id', subjects)]

    # to iterate over sessions of each subject
    infosrc_sessions = Node(IdentityInterface(fields=['session_id']), name="infosrc_sessions")
    infosrc_sessions.itersource = ("infosrc_subjects", 'subject_id')
    infosrc_sessions.iterables = [('session_id', sessions)]

    # File selector (to list files for the pipeline based on the info sources)
    anat_file = os.path.join('{subject_id}_ses-01_run-01_T1w' + f'{args.anat_scan_suffix}.nii')
    func_file = os.path.join('{subject_id}', '{session_id}', 'func', '*bold.nii.gz')

    selectfiles_anat = Node(
        SelectFiles({'anat': anat_file}, base_directory=args.downsampled_anat_data_dir), name="selectfiles_anat"
    )

    selectfiles_sessions = Node(
        SelectFiles({'func': func_file}, base_directory=args.bids_data_dir), name="selectfiles_sessions"
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

    # connect segment
    preproc.connect([(selectfiles_anat, segment_node, [('anat', 'channel_files')])])

    # keeping realignment params
    preproc.connect([(realign_node, datasink_node, [('realignment_parameters', 'realignment.@par')])])

    # preproc.connect([(normalize, datasink_node, [('normalized_files', 'normalized.@files')])])
    preproc.connect([(coregister_node, datasink_node, [('coregistered_files', 'coregistered.@files')])])

    preproc.connect([(segment_node, datasink_node, [('native_class_images', 'segmented.@image')])])

    # draw graph of the pipeline
    preproc.write_graph(graph2use='flat', format='png', simple_form=True)

    # run thef pipeline
    preproc.run('MultiProc', plugin_args={'n_procs': 15})


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bids-data-dir", type=str, default=FMRI_BIDS_DATA_DIR)
    parser.add_argument("--out-data-dir", type=str, default=FMRI_PREPROCESSED_DATA_DIR)

    parser.add_argument("--downsampled-anat-data-dir", type=str, default=FMRI_DOWNSAMPLED_ANAT_DATA_DIR)
    parser.add_argument("--anat-scan-suffix", type=str, default=DEFAULT_ANAT_SCAN_SUFFIX)

    parser.add_argument("--subjects", type=str, nargs='+', default=SUBJECTS)

    parser.add_argument("--sessions", type=str, nargs='+', default=None,
                        help="Default value of None uses all sessions")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    run(args)
