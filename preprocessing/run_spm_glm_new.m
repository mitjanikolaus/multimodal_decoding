function []=run_spm_glm_stage_1(subject)
    addpath('~/apps/spm12');
    spm('Defaults', 'fMRI');
    setenv('SPM_HTML_BROWSER','0');
    spm_jobman('initcfg');
    spm_get_defaults('cmdline',true);

    % increase maximum RAM and keep temporary GLM files in memory
    global defaults
    defaults.stats.maxmem = 2^34;
    defaults.stats.resmem = true;

    maxNumCompThreads

    home = getenv('HOME');
    data_dir = [home,'/data/multimodal_decoding/fmri/betas/', subject, '/unstructured'];
    cd(data_dir)

    %design
    load spm_job;
    spm_jobman('run', jobs);


    % concatenate runs
    % nscans = [2496 2496 2496 2496 2496 2496 2496 2496 2496 2496 2496];
    % load SPM.mat
    % spm_fmri_concatenate('SPM.mat', nscans);


    % glm
    clearvars -except data_dir

    % save residuals
    matlabbatch{1}.spm.stats.fmri_est.spmmat = {[data_dir '/SPM.mat']};
    matlabbatch{1}.spm.stats.fmri_est.write_residuals = 1;
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