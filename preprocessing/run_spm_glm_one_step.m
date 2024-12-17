function []=run_spm_glm_stage_1(subject)
    addpath('~/apps/spm12');
    spm('Defaults', 'fMRI');
    setenv('SPM_HTML_BROWSER','0');
    spm_jobman('initcfg');
    spm_get_defaults('cmdline',true);

    home = getenv('HOME');
    data_dir = [home,'/data/multimodal_decoding/fmri/betas_one_step/', subject, '/unstructured'];
    cd(data_dir)

    %design
    load spm_job;
    spm_jobman('run', jobs);

    % glm
    clearvars -except data_dir

    % increase maximum RAM and keep temporary GLM files in memory
    global defaults
    defaults.stats.maxmem = 2^35;
    defaults.stats.resmem = false;

    % do not save residuals
    matlabbatch{1}.spm.stats.fmri_est.spmmat = {[data_dir '/SPM.mat']};
    matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;

    addpath('~/apps/spm12');
    spm('Defaults', 'fMRI');
    setenv('SPM_HTML_BROWSER','0');
    spm_jobman('initcfg');
    spm_get_defaults('cmdline',true);

    maxNumCompThreads

    defaults.stats.maxmem

    spm_jobman('run', matlabbatch);
end