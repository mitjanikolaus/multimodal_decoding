home = getenv('HOME');
subject = 'sub-01'
filelist = dir(fullfile(home, '/data/multimodal_decoding/fmri/betas/', subject, '/unstructured/run_*'));  %get list of files and folders in any subfolder
filelist = filelist([filelist.isdir]);

for i = 1:numel(filelist)
    clearvars -except i filelist
    data_dir = [filelist(i).folder, '/', filelist(i).name];
    i
    data_dir

    addpath('~/apps/spm12');
    spm('Defaults', 'fMRI');
    setenv('SPM_HTML_BROWSER','0');
    spm_jobman('initcfg');
    spm_get_defaults('cmdline',true);

    cd(data_dir)

    %design
    load spm_lvl1_job_stage_2;
    spm_jobman('run', jobs);


    % concatenate runs
    % nscans = [2496 2496 2496 2496 2496 2496 2496 2496 2496 2496 2496];
    % load SPM.mat
    % spm_fmri_concatenate('SPM.mat', nscans);


    % glm
    clearvars -except data_dir i filelist

    % save residuals
    matlabbatch{1}.spm.stats.fmri_est.spmmat = {[data_dir '/SPM.mat']};
    matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;

    addpath('~/apps/spm12');
    spm('Defaults', 'fMRI');
    setenv('SPM_HTML_BROWSER','0');
    spm_jobman('initcfg');
    spm_get_defaults('cmdline',true);

    spm_jobman('run', matlabbatch);
    % load SPM.mat;
    % spm_spm(SPM);
end