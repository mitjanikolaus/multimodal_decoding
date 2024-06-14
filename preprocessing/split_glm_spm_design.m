clearvars
addpath('~/apps/spm12');
spm('Defaults', 'fMRI');
setenv('SPM_HTML_BROWSER','0');
spm_jobman('initcfg');
spm_get_defaults('cmdline',true);

data_dir = '~/data/multimodal_decoding/fmri/betas/sub-01/unstructured';
cd(data_dir)

%design
load spm_lvl1_job_stage_1;
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
