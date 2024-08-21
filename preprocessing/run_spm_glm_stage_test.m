home = getenv('HOME');
filelist = dir(fullfile(home, '/data/multimodal_decoding/fmri/betas/sub-01/unstructured/run_*'));  %get list of files and folders in any subfolder
filelist = filelist([filelist.isdir]);

for i = 1:numel(filelist)
    clearvars -except i
    data_dir = [filelist(i).folder, '/', filelist(i).name];
    i
    data_dir
    addpath('~/apps/spm12');
    spm('Defaults', 'fMRI');
    setenv('SPM_HTML_BROWSER','0');
    spm_jobman('initcfg');
    spm_get_defaults('cmdline',true);

    cd(data_dir)
    print(data_dir)

    %design
    load spm_lvl1_job_stage_2;
end