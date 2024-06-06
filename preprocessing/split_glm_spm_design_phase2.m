for i=1:153
    i
    clearvars -except i
    addpath('/home/milad/apps/spm12');
    spm('Defaults', 'fMRI');
    setenv('SPM_HTML_BROWSER','0');
    spm_jobman('initcfg');
    spm_get_defaults('cmdline',true);
    
    run_name = sprintf('run_%03d', i);
    data_dir = ['/mnt/HD1/milad/multimodal_decoding/glm_manual/two-stage/sub-07/unstructured/' run_name];
    data_dir
    cd(data_dir)

    %design
    load spm_lvl1_job_stage_2;
    spm_jobman('run', jobs);


    % concatenate runs
    % nscans = [2496 2496 2496 2496 2496 2496 2496 2496 2496 2496 2496];
    % load SPM.mat
    % spm_fmri_concatenate('SPM.mat', nscans);


    % glm
    clearvars -except data_dir i

    % save residuals
    matlabbatch{1}.spm.stats.fmri_est.spmmat = {[data_dir '/SPM.mat']};
    matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;

    addpath('/home/milad/apps/spm12');
    spm('Defaults', 'fMRI');
    setenv('SPM_HTML_BROWSER','0');
    spm_jobman('initcfg');
    spm_get_defaults('cmdline',true);

    spm_jobman('run', matlabbatch);
    % load SPM.mat;
    % spm_spm(SPM);
end