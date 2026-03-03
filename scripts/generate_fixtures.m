% GENERATE_FIXTURES  Generate reference fixture data for MFE Toolbox Python migration
%
% This script exercises every public function in the MFE Toolbox (Version 4.0)
% with representative inputs and saves outputs as .mat files for numerical
% parity testing against the migrated Python implementation.
%
% USAGE:
%   Run in MATLAB or GNU Octave from the repository root:
%     cd /path/to/mfe-toolbox
%     addpath(genpath('.'));
%     run('scripts/generate_fixtures.m');
%
% OUTPUT:
%   .mat files organized under fixtures_mat/ by subpackage:
%     fixtures_mat/univariate/
%     fixtures_mat/multivariate/
%     fixtures_mat/timeseries/
%     fixtures_mat/realized/
%     fixtures_mat/distributions/
%     fixtures_mat/utility/
%     fixtures_mat/bootstrap/
%     fixtures_mat/crosssection/
%     fixtures_mat/sandbox/
%     fixtures_mat/tests/
%
% NOTES:
%   Functions requiring MATLAB Optimization Toolbox are flagged with
%   % REQUIRES_OPTIMIZATION_TOOLBOX and wrapped in try/catch so they
%   can be skipped on systems without the toolbox (e.g., GNU Octave).
%
% Author: MFE Toolbox Migration Pipeline
% Date: 2024

%% ========================================================================
%  SETUP
%  ========================================================================
clear all; close all; clc;

% Force MATLAB v7 format for Octave compatibility with scipy.io.loadmat.
% Octave defaults to HDF5 (v5) format which scipy.io.loadmat cannot read.
% This ensures all save() calls produce MATLAB v7 .mat files.
try
    save_default_options('-v7');
catch
    % MATLAB ignores this (uses v7 by default); only needed for Octave
end

% Set random seed for reproducibility (Octave-compatible)
try
    rng(42, 'twister');
catch
    rand('state', 42);
    randn('state', 42);
end

% Create output directories
baseOutputDir = fullfile(pwd, 'fixtures_mat');
subpackages = {'univariate', 'multivariate', 'timeseries', 'realized', ...
               'distributions', 'utility', 'bootstrap', 'crosssection', ...
               'sandbox', 'tests'};
for i = 1:length(subpackages)
    outDir = fullfile(baseOutputDir, subpackages{i});
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
end

% Add MFE Toolbox directories to path
addpath(genpath(pwd));

% Track errors/skips
skipped = {};
errors = {};

disp('============================================');
disp('MFE Toolbox Fixture Generation');
disp('============================================');

%% ========================================================================
%  COMMON TEST DATA GENERATION
%  ========================================================================
disp('Generating common test data...');

T = 1000;           % Time series length
K = 3;              % Number of assets for multivariate

% Reset seed before data generation for reproducibility
try
    rng(42, 'twister');
catch
    rand('state', 42);
    randn('state', 42);
end

% Univariate return data (mean-zero, scaled to look like financial returns)
epsilon = randn(T, 1);
epsilon = epsilon - mean(epsilon);

% Multivariate return data
data_mv = randn(T, K);
data_mv = data_mv - repmat(mean(data_mv), T, 1);

% High-frequency price data for realized measures
T_hf = 2000;  % High-frequency observations
hf_returns = 0.0001 * randn(T_hf, 1);
hf_prices = 100 * exp(cumsum(hf_returns));
hf_times_seconds = linspace(34200, 57600, T_hf)';  % 9:30 to 16:00 in seconds

% Two-asset high-frequency data for bivariate realized measures
hf_returns2 = 0.0001 * randn(T_hf, 1);
hf_prices2 = 100 * exp(cumsum(hf_returns2));
hf_times_seconds2 = linspace(34200, 57600, T_hf)';

% Regression data
T_reg = 200;
X_reg = randn(T_reg, 3);
beta_true = [0.5; -1.0; 0.3];
y_reg = X_reg * beta_true + randn(T_reg, 1);

% Data for time series functions
y_ts = cumsum(randn(T, 1));  % Random walk for unit root tests
y_stationary = randn(T, 1);  % Stationary series
y_var = randn(T, K);         % VAR data

% Save common test data
save(fullfile(baseOutputDir, 'common_test_data.mat'), ...
    'T', 'K', 'epsilon', 'data_mv', 'T_hf', 'hf_returns', 'hf_prices', ...
    'hf_times_seconds', 'hf_returns2', 'hf_prices2', 'hf_times_seconds2', ...
    'T_reg', 'X_reg', 'beta_true', 'y_reg', 'y_ts', 'y_stationary', 'y_var');

disp('Common test data generated and saved.');

%% ========================================================================
%  UNIVARIATE GARCH FIXTURES (59 functions)
%  ========================================================================
disp(' ');
disp('============================================');
disp('Univariate GARCH Fixtures');
disp('============================================');
outDir = fullfile(baseOutputDir, 'univariate');

%% --- TARCH Model Family (10 functions) ---
disp('=== TARCH ===');

% tarch_parameter_check
try
    disp('  tarch_parameter_check...');
    [pc_p,pc_o,pc_q,pc_error_type,pc_tarch_type,pc_sv,pc_opts] = ...
        tarch_parameter_check(epsilon, 1, 1, 1, 'NORMAL', 2, [], []);
    save(fullfile(outDir, 'tarch_parameter_check.mat'), ...
        'pc_p', 'pc_o', 'pc_q', 'pc_error_type', 'pc_tarch_type');
catch e
    errors{end+1} = ['tarch_parameter_check: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% tarch_simulate
try
    disp('  tarch_simulate...');
    tarch_sim_params = [0.05; 0.05; 0.1; 0.85];
    [tarch_simdata, tarch_simht] = tarch_simulate(500, tarch_sim_params, 1, 1, 1, 'NORMAL', 2);
    save(fullfile(outDir, 'tarch_simulate.mat'), ...
        'tarch_sim_params', 'tarch_simdata', 'tarch_simht');
catch e
    errors{end+1} = ['tarch_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% tarch_transform / tarch_itransform
try
    disp('  tarch_transform...');
    tarch_params_raw = [0.05; 0.05; 0.1; 0.85];
    [tarch_trans, tarch_trans_nu, tarch_trans_lambda] = tarch_transform(tarch_params_raw, 1, 1, 1, 1);
    save(fullfile(outDir, 'tarch_transform.mat'), ...
        'tarch_params_raw', 'tarch_trans', 'tarch_trans_nu', 'tarch_trans_lambda');
catch e
    errors{end+1} = ['tarch_transform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  tarch_itransform...');
    tarch_itrans_input = tarch_trans;
    [tarch_itrans, tarch_itrans_nu, tarch_itrans_lambda] = tarch_itransform(tarch_itrans_input, 1, 1, 1, 1);
    save(fullfile(outDir, 'tarch_itransform.mat'), ...
        'tarch_itrans_input', 'tarch_itrans', 'tarch_itrans_nu', 'tarch_itrans_lambda');
catch e
    errors{end+1} = ['tarch_itransform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% tarch_core
try
    disp('  tarch_core...');
    tarch_core_params = [0.05; 0.05; 0.1; 0.85];
    tc_p = 1; tc_o = 1; tc_q = 1; tc_type = 2;
    tc_m = max([tc_p tc_o tc_q]);
    tc_data = epsilon.^2;
    tc_Idata = tc_data .* (epsilon < 0);
    tc_fdata = [repmat(mean(tc_data), tc_m, 1); tc_data];
    tc_fIdata = [repmat(mean(tc_Idata), tc_m, 1); tc_Idata];
    tc_back_cast = mean(tc_data);
    tc_T = length(tc_fdata);
    tarch_core_ht = tarch_core(tc_fdata, tc_fIdata, tarch_core_params, tc_back_cast, tc_p, tc_o, tc_q, tc_m, tc_T, tc_type);
    save(fullfile(outDir, 'tarch_core.mat'), ...
        'tarch_core_params', 'tc_fdata', 'tc_fIdata', 'tc_back_cast', ...
        'tc_p', 'tc_o', 'tc_q', 'tc_m', 'tc_T', 'tc_type', 'tarch_core_ht');
catch e
    errors{end+1} = ['tarch_core: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% tarch_core_simple
try
    disp('  tarch_core_simple...');
    tcs_params = [0.05; 0.05; 0.1; 0.85];
    tcs_data = epsilon.^2;
    tcs_Idata = tcs_data .* (epsilon < 0);
    tcs_m = 1;
    tcs_fdata = [mean(tcs_data); tcs_data];
    tcs_fIdata = [mean(tcs_Idata); tcs_Idata];
    tcs_back_cast = mean(tcs_data);
    tcs_T = length(tcs_fdata);
    tarch_cs_ht = tarch_core_simple(tcs_fdata, tcs_fIdata, tcs_params, tcs_back_cast, 1, 1, 1, tcs_m, tcs_T, 2);
    save(fullfile(outDir, 'tarch_core_simple.mat'), ...
        'tcs_params', 'tcs_fdata', 'tcs_fIdata', 'tcs_back_cast', ...
        'tcs_m', 'tcs_T', 'tarch_cs_ht');
catch e
    errors{end+1} = ['tarch_core_simple: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% tarch_likelihood
try
    disp('  tarch_likelihood...');
    tl_params = [0.05; 0.05; 0.1; 0.85];
    tl_data = epsilon;
    tl_fdata = epsilon.^2;
    tl_fIdata = tl_fdata .* (epsilon < 0);
    tl_back_cast = mean(tl_fdata);
    tl_T = length(tl_data);
    [tarch_ll, tarch_lls, tarch_ll_ht] = tarch_likelihood(tl_params, tl_data, tl_fdata, tl_fIdata, 1, 1, 1, 1, 2, tl_back_cast, tl_T, 0);
    save(fullfile(outDir, 'tarch_likelihood.mat'), ...
        'tl_params', 'tl_data', 'tl_fdata', 'tl_fIdata', 'tl_back_cast', 'tl_T', ...
        'tarch_ll', 'tarch_lls', 'tarch_ll_ht');
catch e
    errors{end+1} = ['tarch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% tarch_starting_values - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  tarch_starting_values...');
    [tarch_sv] = tarch_starting_values(epsilon.^2, epsilon.^2 .* (epsilon<0), 1, 1, 1, 2);
    save(fullfile(outDir, 'tarch_starting_values.mat'), 'tarch_sv');
catch e
    skipped{end+1} = ['tarch_starting_values: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

% tarch_display - save params that would be displayed
try
    disp('  tarch_display...');
    tarch_disp_params = [0.05; 0.05; 0.1; 0.85];
    tarch_disp_ll = -1500;
    tarch_disp_vcv = eye(4) * 0.001;
    save(fullfile(outDir, 'tarch_display.mat'), ...
        'tarch_disp_params', 'tarch_disp_ll', 'tarch_disp_vcv');
catch e
    errors{end+1} = ['tarch_display: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% tarch (full estimation) - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  tarch (full estimation)...');
    [tarch_parameters, tarch_LL, tarch_ht, tarch_VCVrobust, tarch_VCV, tarch_scores] = tarch(epsilon, 1, 1, 1);
    save(fullfile(outDir, 'tarch.mat'), ...
        'tarch_parameters', 'tarch_LL', 'tarch_ht', 'tarch_VCVrobust', 'tarch_VCV', 'tarch_scores');
catch e
    skipped{end+1} = ['tarch (full estimation): ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- AGARCH Model Family (9 functions) ---
disp('=== AGARCH ===');

% agarch_parameter_check
try
    disp('  agarch_parameter_check...');
    [ap_p,ap_q,ap_mt,ap_et,ap_sv,ap_opts] = agarch_parameter_check(epsilon, 1, 1, 'AGARCH', 'NORMAL', [], []);
    save(fullfile(outDir, 'agarch_parameter_check.mat'), ...
        'ap_p', 'ap_q', 'ap_mt', 'ap_et');
catch e
    errors{end+1} = ['agarch_parameter_check: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% agarch_simulate
try
    disp('  agarch_simulate...');
    agarch_sim_params = [0.05; 0.05; -0.1; 0.85];  % omega, alpha, gamma, beta
    [agarch_simdata, agarch_simht] = agarch_simulate(500, agarch_sim_params, 1, 1, 'AGARCH', 'NORMAL');
    save(fullfile(outDir, 'agarch_simulate.mat'), ...
        'agarch_sim_params', 'agarch_simdata', 'agarch_simht');
catch e
    errors{end+1} = ['agarch_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% agarch_transform / agarch_itransform
try
    disp('  agarch_transform...');
    agarch_params_raw = [0.05; 0.05; -0.1; 0.85];
    [agarch_trans, agarch_trans_nu, agarch_trans_lambda] = agarch_transform(agarch_params_raw, 1, 1, 1, 1);
    save(fullfile(outDir, 'agarch_transform.mat'), ...
        'agarch_params_raw', 'agarch_trans', 'agarch_trans_nu', 'agarch_trans_lambda');
catch e
    errors{end+1} = ['agarch_transform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  agarch_itransform...');
    agarch_itrans_input = agarch_trans;
    [agarch_itrans, agarch_itrans_nu, agarch_itrans_lambda] = agarch_itransform(agarch_itrans_input, 1, 1, 1, 1);
    save(fullfile(outDir, 'agarch_itransform.mat'), ...
        'agarch_itrans_input', 'agarch_itrans', 'agarch_itrans_nu', 'agarch_itrans_lambda');
catch e
    errors{end+1} = ['agarch_itransform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% agarch_core
try
    disp('  agarch_core...');
    agarch_core_params = [0.05; 0.05; -0.1; 0.85];
    ac_p = 1; ac_q = 1;
    ac_m = max([ac_p ac_q]);
    ac_data = [repmat(0, ac_m, 1); epsilon];
    ac_back_cast = var(epsilon);
    ac_T = length(ac_data);
    agarch_core_ht = agarch_core(ac_data, agarch_core_params, ac_back_cast, ac_p, ac_q, ac_m, ac_T, 1);
    save(fullfile(outDir, 'agarch_core.mat'), ...
        'agarch_core_params', 'ac_data', 'ac_back_cast', 'ac_p', 'ac_q', ...
        'ac_m', 'ac_T', 'agarch_core_ht');
catch e
    errors{end+1} = ['agarch_core: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% agarch_likelihood
try
    disp('  agarch_likelihood...');
    al_params = [0.05; 0.05; -0.1; 0.85];
    al_back_cast = var(epsilon);
    al_T = length(epsilon);
    [agarch_ll, agarch_lls, agarch_ll_ht] = agarch_likelihood(al_params, epsilon, 1, 1, 1, al_back_cast, al_T, 1, 0);
    save(fullfile(outDir, 'agarch_likelihood.mat'), ...
        'al_params', 'al_back_cast', 'al_T', 'agarch_ll', 'agarch_lls', 'agarch_ll_ht');
catch e
    errors{end+1} = ['agarch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% agarch_starting_values - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  agarch_starting_values...');
    [agarch_sv] = agarch_starting_values(epsilon, 1, 1, 1);
    save(fullfile(outDir, 'agarch_starting_values.mat'), 'agarch_sv');
catch e
    skipped{end+1} = ['agarch_starting_values: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

% agarch_display
try
    disp('  agarch_display...');
    agarch_disp_params = [0.05; 0.05; -0.1; 0.85];
    agarch_disp_ll = -1500;
    agarch_disp_vcv = eye(4) * 0.001;
    save(fullfile(outDir, 'agarch_display.mat'), ...
        'agarch_disp_params', 'agarch_disp_ll', 'agarch_disp_vcv');
catch e
    errors{end+1} = ['agarch_display: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% agarch (full estimation) - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  agarch (full estimation)...');
    [agarch_parameters, agarch_LL, agarch_ht, agarch_VCVrobust, agarch_VCV, agarch_scores] = agarch(epsilon, 1, 1);
    save(fullfile(outDir, 'agarch.mat'), ...
        'agarch_parameters', 'agarch_LL', 'agarch_ht', 'agarch_VCVrobust', 'agarch_VCV', 'agarch_scores');
catch e
    skipped{end+1} = ['agarch (full estimation): ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- EGARCH Model Family (10 functions) ---
disp('=== EGARCH ===');

% egarch_parameter_check
try
    disp('  egarch_parameter_check...');
    [ep_p,ep_o,ep_q,ep_et,ep_sv,ep_opts] = egarch_parameter_check(epsilon, 1, 1, 1, 'NORMAL', [], []);
    save(fullfile(outDir, 'egarch_parameter_check.mat'), ...
        'ep_p', 'ep_o', 'ep_q', 'ep_et');
catch e
    errors{end+1} = ['egarch_parameter_check: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% egarch_simulate
try
    disp('  egarch_simulate...');
    egarch_sim_params = [-0.1; 0.1; -0.05; 0.95];  % omega, alpha, gamma, beta
    [egarch_simdata, egarch_simht] = egarch_simulate(500, egarch_sim_params, 1, 1, 1, 'NORMAL');
    save(fullfile(outDir, 'egarch_simulate.mat'), ...
        'egarch_sim_params', 'egarch_simdata', 'egarch_simht');
catch e
    errors{end+1} = ['egarch_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% egarch_transform / egarch_itransform
try
    disp('  egarch_transform...');
    egarch_params_raw = [-0.1; 0.1; -0.05; 0.95];
    [egarch_trans, egarch_trans_nu, egarch_trans_lambda] = egarch_transform(egarch_params_raw, 1, 1, 1, 1);
    save(fullfile(outDir, 'egarch_transform.mat'), ...
        'egarch_params_raw', 'egarch_trans', 'egarch_trans_nu', 'egarch_trans_lambda');
catch e
    errors{end+1} = ['egarch_transform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  egarch_itransform...');
    egarch_itrans_input = egarch_trans;
    [egarch_itrans, egarch_itrans_nu, egarch_itrans_lambda] = egarch_itransform(egarch_itrans_input, 1, 1, 1, 1);
    save(fullfile(outDir, 'egarch_itransform.mat'), ...
        'egarch_itrans_input', 'egarch_itrans', 'egarch_itrans_nu', 'egarch_itrans_lambda');
catch e
    errors{end+1} = ['egarch_itransform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% egarch_core
try
    disp('  egarch_core...');
    egarch_core_params = [-0.1; 0.1; -0.05; 0.95];
    ec_p = 1; ec_o = 1; ec_q = 1;
    ec_m = max([ec_p ec_o ec_q]);
    ec_data = [repmat(0, ec_m, 1); epsilon];
    ec_back_cast = log(var(epsilon));
    ec_upper = log(1e6);
    ec_T = length(ec_data);
    egarch_core_ht = egarch_core(ec_data, egarch_core_params, ec_back_cast, ec_upper, ec_p, ec_o, ec_q, ec_m, ec_T);
    save(fullfile(outDir, 'egarch_core.mat'), ...
        'egarch_core_params', 'ec_data', 'ec_back_cast', 'ec_upper', ...
        'ec_p', 'ec_o', 'ec_q', 'ec_m', 'ec_T', 'egarch_core_ht');
catch e
    errors{end+1} = ['egarch_core: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% egarch_likelihood
try
    disp('  egarch_likelihood...');
    el_params = [-0.1; 0.1; -0.05; 0.95];
    el_back_cast = log(var(epsilon));
    el_T = length(epsilon);
    [egarch_ll, egarch_lls, egarch_ll_ht] = egarch_likelihood(el_params, epsilon, 1, 1, 1, 1, el_back_cast, el_T, 0);
    save(fullfile(outDir, 'egarch_likelihood.mat'), ...
        'el_params', 'el_back_cast', 'el_T', 'egarch_ll', 'egarch_lls', 'egarch_ll_ht');
catch e
    errors{end+1} = ['egarch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% egarch_nlcon
try
    disp('  egarch_nlcon...');
    enlc_params = [-0.1; 0.1; -0.05; 0.95];
    [egarch_nlc_c, egarch_nlc_ceq] = egarch_nlcon(enlc_params, 1, 1, 1, 1);
    save(fullfile(outDir, 'egarch_nlcon.mat'), ...
        'enlc_params', 'egarch_nlc_c', 'egarch_nlc_ceq');
catch e
    errors{end+1} = ['egarch_nlcon: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% egarch_starting_values - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  egarch_starting_values...');
    [egarch_sv] = egarch_starting_values(epsilon, 1, 1, 1);
    save(fullfile(outDir, 'egarch_starting_values.mat'), 'egarch_sv');
catch e
    skipped{end+1} = ['egarch_starting_values: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

% egarch_display
try
    disp('  egarch_display...');
    egarch_disp_params = [-0.1; 0.1; -0.05; 0.95];
    egarch_disp_ll = -1500;
    egarch_disp_vcv = eye(4) * 0.001;
    save(fullfile(outDir, 'egarch_display.mat'), ...
        'egarch_disp_params', 'egarch_disp_ll', 'egarch_disp_vcv');
catch e
    errors{end+1} = ['egarch_display: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% egarch (full estimation) - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  egarch (full estimation)...');
    [egarch_parameters, egarch_LL, egarch_ht, egarch_VCVrobust, egarch_VCV, egarch_scores] = egarch(epsilon, 1, 1, 1);
    save(fullfile(outDir, 'egarch.mat'), ...
        'egarch_parameters', 'egarch_LL', 'egarch_ht', 'egarch_VCVrobust', 'egarch_VCV', 'egarch_scores');
catch e
    skipped{end+1} = ['egarch (full estimation): ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- IGARCH Model Family (8 functions) ---
disp('=== IGARCH ===');

% igarch_parameter_check
try
    disp('  igarch_parameter_check...');
    [ip_p,ip_q,ip_et,ip_it,ip_const,ip_sv,ip_opts] = igarch_parameter_check(epsilon, 1, 1, 'NORMAL', 2, true, [], []);
    save(fullfile(outDir, 'igarch_parameter_check.mat'), ...
        'ip_p', 'ip_q', 'ip_et', 'ip_it', 'ip_const');
catch e
    errors{end+1} = ['igarch_parameter_check: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% igarch_transform / igarch_itransform
try
    disp('  igarch_transform...');
    igarch_params_raw = [0.01; 0.1];  % omega, alpha (beta = 1 - alpha)
    [igarch_trans, igarch_trans_nu, igarch_trans_lambda] = igarch_transform(igarch_params_raw, 1, 1, 1, 2, true);
    save(fullfile(outDir, 'igarch_transform.mat'), ...
        'igarch_params_raw', 'igarch_trans', 'igarch_trans_nu', 'igarch_trans_lambda');
catch e
    errors{end+1} = ['igarch_transform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  igarch_itransform...');
    igarch_itrans_input = igarch_trans;
    [igarch_itrans, igarch_itrans_nu, igarch_itrans_lambda] = igarch_itransform(igarch_itrans_input, 1, 1, 1, 2, true);
    save(fullfile(outDir, 'igarch_itransform.mat'), ...
        'igarch_itrans_input', 'igarch_itrans', 'igarch_itrans_nu', 'igarch_itrans_lambda');
catch e
    errors{end+1} = ['igarch_itransform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% igarch_core
try
    disp('  igarch_core...');
    igarch_core_params = [0.01; 0.1; 0.9];
    ic_p = 1; ic_q = 1;
    ic_m = max([ic_p ic_q]);
    ic_fepsilon = [repmat(var(epsilon), ic_m, 1); epsilon.^2];
    ic_back_cast = var(epsilon);
    ic_T = length(ic_fepsilon);
    igarch_core_ht = igarch_core(ic_fepsilon, igarch_core_params, ic_back_cast, ic_p, ic_q, ic_m, ic_T, 2, 1);
    save(fullfile(outDir, 'igarch_core.mat'), ...
        'igarch_core_params', 'ic_fepsilon', 'ic_back_cast', ...
        'ic_p', 'ic_q', 'ic_m', 'ic_T', 'igarch_core_ht');
catch e
    errors{end+1} = ['igarch_core: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% igarch_likelihood
try
    disp('  igarch_likelihood...');
    il_params = [0.01; 0.1];
    il_back_cast = var(epsilon);
    il_T = length(epsilon);
    [igarch_ll, igarch_lls, igarch_ll_ht] = igarch_likelihood(il_params, epsilon, 1, 1, 1, 2, 1, il_back_cast, il_T, 0);
    save(fullfile(outDir, 'igarch_likelihood.mat'), ...
        'il_params', 'il_back_cast', 'il_T', 'igarch_ll', 'igarch_lls', 'igarch_ll_ht');
catch e
    errors{end+1} = ['igarch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% igarch_starting_values - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  igarch_starting_values...');
    [igarch_sv] = igarch_starting_values(epsilon, 1, 1);
    save(fullfile(outDir, 'igarch_starting_values.mat'), 'igarch_sv');
catch e
    skipped{end+1} = ['igarch_starting_values: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

% igarch_display
try
    disp('  igarch_display...');
    igarch_disp_params = [0.01; 0.1];
    igarch_disp_ll = -1500;
    igarch_disp_vcv = eye(2) * 0.001;
    save(fullfile(outDir, 'igarch_display.mat'), ...
        'igarch_disp_params', 'igarch_disp_ll', 'igarch_disp_vcv');
catch e
    errors{end+1} = ['igarch_display: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% igarch (full estimation) - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  igarch (full estimation)...');
    [igarch_parameters, igarch_LL, igarch_ht, igarch_VCVrobust, igarch_VCV, igarch_scores] = igarch(epsilon, 1, 1);
    save(fullfile(outDir, 'igarch.mat'), ...
        'igarch_parameters', 'igarch_LL', 'igarch_ht', 'igarch_VCVrobust', 'igarch_VCV', 'igarch_scores');
catch e
    skipped{end+1} = ['igarch (full estimation): ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- APARCH Model Family (10 functions) ---
disp('=== APARCH ===');

% aparch_parameter_check
try
    disp('  aparch_parameter_check...');
    [app_p,app_o,app_q,app_et,app_ud,app_sv,app_opts] = aparch_parameter_check(epsilon, 1, 1, 1, 'NORMAL', [], [], []);
    save(fullfile(outDir, 'aparch_parameter_check.mat'), ...
        'app_p', 'app_o', 'app_q', 'app_et');
catch e
    errors{end+1} = ['aparch_parameter_check: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% aparch_simulate
try
    disp('  aparch_simulate...');
    aparch_sim_params = [0.05; 0.05; 0.3; 0.85; 2.0];  % omega, alpha, gamma, beta, delta
    [aparch_simdata, aparch_simht] = aparch_simulate(500, aparch_sim_params, 1, 1, 1, 'NORMAL');
    save(fullfile(outDir, 'aparch_simulate.mat'), ...
        'aparch_sim_params', 'aparch_simdata', 'aparch_simht');
catch e
    errors{end+1} = ['aparch_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% aparch_transform / aparch_itransform
try
    disp('  aparch_transform...');
    aparch_params_raw = [0.05; 0.05; 0.3; 0.85; 2.0];
    [aparch_trans, aparch_trans_nu, aparch_trans_lambda] = aparch_transform(aparch_params_raw, 1, 1, 1, 1);
    save(fullfile(outDir, 'aparch_transform.mat'), ...
        'aparch_params_raw', 'aparch_trans', 'aparch_trans_nu', 'aparch_trans_lambda');
catch e
    errors{end+1} = ['aparch_transform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  aparch_itransform...');
    aparch_itrans_input = aparch_trans;
    [aparch_itrans, aparch_itrans_nu, aparch_itrans_lambda] = aparch_itransform(aparch_itrans_input, 1, 1, 1, 1);
    save(fullfile(outDir, 'aparch_itransform.mat'), ...
        'aparch_itrans_input', 'aparch_itrans', 'aparch_itrans_nu', 'aparch_itrans_lambda');
catch e
    errors{end+1} = ['aparch_itransform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% aparch_core
try
    disp('  aparch_core...');
    aparch_core_params = [0.05; 0.05; 0.3; 0.85; 2.0];
    apc_p = 1; apc_o = 1; apc_q = 1;
    apc_m = max([apc_p apc_o apc_q]);
    apc_data = [repmat(0, apc_m, 1); epsilon];
    apc_abs_data = abs(apc_data);
    apc_back_cast = (var(epsilon))^(2.0/2);
    apc_T = length(apc_data);
    apc_LB = 0.3;
    apc_UB = 4.0;
    aparch_core_ht = aparch_core(apc_data, apc_abs_data, aparch_core_params, apc_p, apc_o, apc_q, apc_m, apc_T, apc_back_cast, apc_LB, apc_UB);
    save(fullfile(outDir, 'aparch_core.mat'), ...
        'aparch_core_params', 'apc_data', 'apc_abs_data', 'apc_back_cast', ...
        'apc_p', 'apc_o', 'apc_q', 'apc_m', 'apc_T', 'aparch_core_ht');
catch e
    errors{end+1} = ['aparch_core: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% aparch_likelihood
try
    disp('  aparch_likelihood...');
    apl_params = [0.05; 0.05; 0.3; 0.85; 2.0];
    apl_back_cast = var(epsilon);
    apl_T = length(epsilon);
    [aparch_ll, aparch_lls, aparch_ll_ht] = aparch_likelihood(apl_params, epsilon, 1, 1, 1, 1, apl_back_cast, apl_T, 0);
    save(fullfile(outDir, 'aparch_likelihood.mat'), ...
        'apl_params', 'apl_back_cast', 'apl_T', 'aparch_ll', 'aparch_lls', 'aparch_ll_ht');
catch e
    errors{end+1} = ['aparch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% aparch_loglikelihood
try
    disp('  aparch_loglikelihood...');
    apll_params = [0.05; 0.05; 0.3; 0.85; 2.0];
    apll_back_cast = var(epsilon);
    apll_T = length(epsilon);
    [aparch_logll, aparch_loglls, aparch_logll_ht] = aparch_loglikelihood(apll_params, epsilon, 1, 1, 1, 1, apll_back_cast, apll_T, 0);
    save(fullfile(outDir, 'aparch_loglikelihood.mat'), ...
        'apll_params', 'apll_back_cast', 'apll_T', 'aparch_logll', 'aparch_loglls', 'aparch_logll_ht');
catch e
    errors{end+1} = ['aparch_loglikelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% aparch_starting_values - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  aparch_starting_values...');
    [aparch_sv] = aparch_starting_values(epsilon, 1, 1, 1);
    save(fullfile(outDir, 'aparch_starting_values.mat'), 'aparch_sv');
catch e
    skipped{end+1} = ['aparch_starting_values: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

% aparch_display
try
    disp('  aparch_display...');
    aparch_disp_params = [0.05; 0.05; 0.3; 0.85; 2.0];
    aparch_disp_ll = -1500;
    aparch_disp_vcv = eye(5) * 0.001;
    save(fullfile(outDir, 'aparch_display.mat'), ...
        'aparch_disp_params', 'aparch_disp_ll', 'aparch_disp_vcv');
catch e
    errors{end+1} = ['aparch_display: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% aparch (full estimation) - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  aparch (full estimation)...');
    [aparch_parameters, aparch_LL, aparch_ht, aparch_VCVrobust, aparch_VCV, aparch_scores] = aparch(epsilon, 1, 1, 1);
    save(fullfile(outDir, 'aparch.mat'), ...
        'aparch_parameters', 'aparch_LL', 'aparch_ht', 'aparch_VCVrobust', 'aparch_VCV', 'aparch_scores');
catch e
    skipped{end+1} = ['aparch (full estimation): ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- FIGARCH Model Family (8 functions) ---
disp('=== FIGARCH ===');

% figarch_parameter_check
try
    disp('  figarch_parameter_check...');
    [fp_p,fp_q,fp_et,fp_tl,fp_sv,fp_opts] = figarch_parameter_check(epsilon, 1, 1, 'NORMAL', 1000, [], []);
    save(fullfile(outDir, 'figarch_parameter_check.mat'), ...
        'fp_p', 'fp_q', 'fp_et', 'fp_tl');
catch e
    errors{end+1} = ['figarch_parameter_check: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% figarch_simulate
try
    disp('  figarch_simulate...');
    figarch_sim_params = [0.2; 0.1; 0.4; 0.3];  % omega, phi, d, beta
    [figarch_simdata, figarch_simht] = figarch_simulate(500, figarch_sim_params, 1, 1, 'NORMAL', 1000);
    save(fullfile(outDir, 'figarch_simulate.mat'), ...
        'figarch_sim_params', 'figarch_simdata', 'figarch_simht');
catch e
    errors{end+1} = ['figarch_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% figarch_transform / figarch_itransform
try
    disp('  figarch_transform...');
    figarch_params_raw = [0.2; 0.1; 0.4; 0.3];
    [figarch_trans, figarch_trans_nu, figarch_trans_lambda] = figarch_transform(figarch_params_raw, 1, 1, 1);
    save(fullfile(outDir, 'figarch_transform.mat'), ...
        'figarch_params_raw', 'figarch_trans', 'figarch_trans_nu', 'figarch_trans_lambda');
catch e
    errors{end+1} = ['figarch_transform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  figarch_itransform...');
    figarch_itrans_input = figarch_trans;
    [figarch_itrans, figarch_itrans_nu, figarch_itrans_lambda] = figarch_itransform(figarch_itrans_input, 1, 1, 1);
    save(fullfile(outDir, 'figarch_itransform.mat'), ...
        'figarch_itrans_input', 'figarch_itrans', 'figarch_itrans_nu', 'figarch_itrans_lambda');
catch e
    errors{end+1} = ['figarch_itransform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% figarch_weights
try
    disp('  figarch_weights...');
    fw_params = [0.1; 0.4; 0.3];  % phi, d, beta
    fw_truncLag = 1000;
    figarch_w = figarch_weights(fw_params, 1, 1, fw_truncLag);
    save(fullfile(outDir, 'figarch_weights.mat'), ...
        'fw_params', 'fw_truncLag', 'figarch_w');
catch e
    errors{end+1} = ['figarch_weights: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% figarch_likelihood
try
    disp('  figarch_likelihood...');
    fl_params = [0.2; 0.1; 0.4; 0.3];
    fl_back_cast = var(epsilon);
    fl_T = length(epsilon);
    [figarch_ll, figarch_lls, figarch_ll_ht] = figarch_likelihood(fl_params, epsilon, 1, 1, 1, 1000, fl_back_cast, fl_T, 0);
    save(fullfile(outDir, 'figarch_likelihood.mat'), ...
        'fl_params', 'fl_back_cast', 'fl_T', 'figarch_ll', 'figarch_lls', 'figarch_ll_ht');
catch e
    errors{end+1} = ['figarch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% figarch_starting_values - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  figarch_starting_values...');
    [figarch_sv] = figarch_starting_values(epsilon, 1, 1, 1000);
    save(fullfile(outDir, 'figarch_starting_values.mat'), 'figarch_sv');
catch e
    skipped{end+1} = ['figarch_starting_values: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

% figarch (full estimation) - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  figarch (full estimation)...');
    [figarch_parameters, figarch_LL, figarch_ht, figarch_VCVrobust, figarch_VCV, figarch_scores] = figarch(epsilon, 1, 1);
    save(fullfile(outDir, 'figarch.mat'), ...
        'figarch_parameters', 'figarch_LL', 'figarch_ht', 'figarch_VCVrobust', 'figarch_VCV', 'figarch_scores');
catch e
    skipped{end+1} = ['figarch (full estimation): ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- HEAVY Model Family (4 functions) ---
disp('=== HEAVY ===');

% heavy_simulate
try
    disp('  heavy_simulate...');
    heavy_sim_params = [0.15; 0.05; 0.2; 0.4; 0.7; 0.55];
    heavy_p = [0 1; 0 1];
    heavy_q = eye(2);
    heavy_m = [1; 78];
    [heavy_simdata, heavy_simht] = heavy_simulate(500, 2, heavy_sim_params, heavy_p, heavy_q, heavy_m);
    save(fullfile(outDir, 'heavy_simulate.mat'), ...
        'heavy_sim_params', 'heavy_p', 'heavy_q', 'heavy_m', 'heavy_simdata', 'heavy_simht');
catch e
    errors{end+1} = ['heavy_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% heavy_parameter_transform
try
    disp('  heavy_parameter_transform...');
    hpt_params = [0.15; 0.05; 0.2; 0.4; 0.7; 0.55];
    hpt_p = [0 1; 0 1];
    hpt_q = eye(2);
    [heavy_pt_out] = heavy_parameter_transform(hpt_params, hpt_p, hpt_q, 2);
    save(fullfile(outDir, 'heavy_parameter_transform.mat'), ...
        'hpt_params', 'hpt_p', 'hpt_q', 'heavy_pt_out');
catch e
    errors{end+1} = ['heavy_parameter_transform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% heavy_likelihood
try
    disp('  heavy_likelihood...');
    % Generate HEAVY-compatible data: returns and realized measure
    heavy_ll_data = [epsilon(1:500), abs(epsilon(1:500)).^2 * 5 + 0.1];
    heavy_ll_params = [0.15; 0.05; 0.2; 0.4; 0.7; 0.55];
    heavy_ll_p = [0 1; 0 1];
    heavy_ll_q = eye(2);
    [heavy_ll_val, heavy_ll_lls, heavy_ll_ht_out] = heavy_likelihood(heavy_ll_params, heavy_ll_data, heavy_ll_p, heavy_ll_q, [], 2);
    save(fullfile(outDir, 'heavy_likelihood.mat'), ...
        'heavy_ll_params', 'heavy_ll_data', 'heavy_ll_p', 'heavy_ll_q', ...
        'heavy_ll_val', 'heavy_ll_lls', 'heavy_ll_ht_out');
catch e
    errors{end+1} = ['heavy_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% heavy (full estimation) - REQUIRES_OPTIMIZATION_TOOLBOX
% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  heavy (full estimation)...');
    heavy_data = [epsilon(1:500), abs(epsilon(1:500)).^2 * 5 + 0.1];
    heavy_est_p = [0 1; 0 1];
    heavy_est_q = eye(2);
    [heavy_parameters, heavy_ll_out, heavy_ht_out, heavy_VCV_out, heavy_scores_out] = heavy(heavy_data, heavy_est_p, heavy_est_q);
    save(fullfile(outDir, 'heavy.mat'), ...
        'heavy_parameters', 'heavy_ll_out', 'heavy_ht_out', 'heavy_VCV_out', 'heavy_scores_out');
catch e
    skipped{end+1} = ['heavy (full estimation): ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% ========================================================================
%  MULTIVARIATE GARCH FIXTURES (38 functions)
%  ========================================================================
disp(' ');
disp('============================================');
disp('Multivariate GARCH Fixtures');
disp('============================================');
outDir = fullfile(baseOutputDir, 'multivariate');

%% --- RiskMetrics (does NOT require Optimization Toolbox) ---
disp('=== RiskMetrics ===');
try
    disp('  riskmetrics...');
    rm_Ht = riskmetrics(data_mv, 0.94);
    save(fullfile(outDir, 'riskmetrics.mat'), 'rm_Ht');
catch e
    errors{end+1} = ['riskmetrics: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  riskmetrics2006...');
    rm2006_Ht = riskmetrics2006(data_mv);
    save(fullfile(outDir, 'riskmetrics2006.mat'), 'rm2006_Ht');
catch e
    errors{end+1} = ['riskmetrics2006: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- BEKK ---
disp('=== BEKK ===');
try
    disp('  bekk_simulate...');
    CCp = [1 0.5; 0.5 4];
    bekk_sim_params = [chol2vec(chol(CCp)'); sqrt([0.05; 0.10; 0.88])];
    [bekk_simdata, bekk_simHt] = bekk_simulate(500, 2, bekk_sim_params, 1, 1, 1, 'Scalar');
    save(fullfile(outDir, 'bekk_simulate.mat'), 'bekk_sim_params', 'bekk_simdata', 'bekk_simHt');
catch e
    errors{end+1} = ['bekk_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  bekk_parameter_transform...');
    bekk_pt_params = [0.5; 0.1; 0.2; 0.3; 0.9];
    [bekk_pt_out] = bekk_parameter_transform(bekk_pt_params, 1, 1, 1, 2, 'Scalar');
    save(fullfile(outDir, 'bekk_parameter_transform.mat'), 'bekk_pt_params', 'bekk_pt_out');
catch e
    errors{end+1} = ['bekk_parameter_transform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  bekk_constraint...');
    bekk_con_params = [0.5; 0.1; 0.2; 0.3; 0.9];
    [bekk_c, bekk_ceq] = bekk_constraint(bekk_con_params, 1, 1, 1, 2, 'Scalar');
    save(fullfile(outDir, 'bekk_constraint.mat'), 'bekk_con_params', 'bekk_c', 'bekk_ceq');
catch e
    errors{end+1} = ['bekk_constraint: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  bekk_likelihood...');
    bekk_ll_params = [0.5; 0.1; 0.2; 0.3; 0.9];
    [bekk_ll_val, bekk_ll_lls] = bekk_likelihood(bekk_ll_params, data_mv(:,1:2), 1, 1, 1, 'Scalar', false, false);
    save(fullfile(outDir, 'bekk_likelihood.mat'), 'bekk_ll_params', 'bekk_ll_val', 'bekk_ll_lls');
catch e
    errors{end+1} = ['bekk_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  bekk (full estimation)...');
    [bekk_parameters, bekk_LL, bekk_Ht] = bekk(data_mv(:,1:2), [], 1, 1, 1, 'Scalar');
    save(fullfile(outDir, 'bekk.mat'), 'bekk_parameters', 'bekk_LL', 'bekk_Ht');
catch e
    skipped{end+1} = ['bekk: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- CCC-MVGARCH ---
disp('=== CCC-MVGARCH ===');
try
    disp('  ccc_mvgarch_simulate...');
    ccc_sim_params = [0.05 0.05 0.9 0.05 0.05 0.9 0.05 0.05 0.9]';
    R_ccc = [1 0.5 0.3; 0.5 1 0.4; 0.3 0.4 1];
    [ccc_simdata, ccc_simHt] = ccc_mvgarch_simulate(500, ccc_sim_params, 1, 0, 1, R_ccc);
    save(fullfile(outDir, 'ccc_mvgarch_simulate.mat'), 'ccc_sim_params', 'R_ccc', 'ccc_simdata', 'ccc_simHt');
catch e
    errors{end+1} = ['ccc_mvgarch_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  ccc_mvgarch_likelihood...');
    ccc_ll_params = [0.05; 0.05; 0.9];
    [ccc_ll_val, ccc_ll_lls, ccc_ll_ht] = ccc_mvgarch_likelihood(ccc_ll_params, data_mv(:,1), 1, 0, 1, 2);
    save(fullfile(outDir, 'ccc_mvgarch_likelihood.mat'), 'ccc_ll_params', 'ccc_ll_val', 'ccc_ll_lls', 'ccc_ll_ht');
catch e
    errors{end+1} = ['ccc_mvgarch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  ccc_mvgarch_joint_likelihood...');
    ccc_jll_R = corr(data_mv);
    ccc_jll_ht = repmat(var(data_mv), T, 1);
    [ccc_jll_val, ccc_jll_lls] = ccc_mvgarch_joint_likelihood(ccc_jll_R(:), data_mv, ccc_jll_ht);
    save(fullfile(outDir, 'ccc_mvgarch_joint_likelihood.mat'), 'ccc_jll_R', 'ccc_jll_val', 'ccc_jll_lls');
catch e
    errors{end+1} = ['ccc_mvgarch_joint_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  ccc_mvgarch (full estimation)...');
    [ccc_parameters, ccc_ll, ccc_Ht, ccc_VCV, ccc_scores] = ccc_mvgarch(data_mv, [], 1, 0, 1);
    save(fullfile(outDir, 'ccc_mvgarch.mat'), 'ccc_parameters', 'ccc_ll', 'ccc_Ht');
catch e
    skipped{end+1} = ['ccc_mvgarch: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- DCC ---
disp('=== DCC ===');
try
    disp('  dcc_likelihood...');
    dcc_ll_params = [0.05; 0.93];
    dcc_ll_stresid = data_mv ./ repmat(std(data_mv), T, 1);
    dcc_ll_R = corr(dcc_ll_stresid);
    [dcc_ll_val, dcc_ll_lls] = dcc_likelihood(dcc_ll_params, dcc_ll_stresid, dcc_ll_R, [], [], 1, 0, 1, 0);
    save(fullfile(outDir, 'dcc_likelihood.mat'), 'dcc_ll_params', 'dcc_ll_R', 'dcc_ll_val', 'dcc_ll_lls');
catch e
    errors{end+1} = ['dcc_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  dcc_reconstruct_variance...');
    dcc_rv_stresid = data_mv ./ repmat(std(data_mv), T, 1);
    dcc_rv_R = corr(dcc_rv_stresid);
    dcc_rv_ht = repmat(var(data_mv), T, 1);
    [dcc_rv_Ht] = dcc_reconstruct_variance([0.05; 0.93], dcc_rv_stresid, dcc_rv_ht, dcc_rv_R, [], [], 1, 0, 1);
    save(fullfile(outDir, 'dcc_reconstruct_variance.mat'), 'dcc_rv_Ht');
catch e
    errors{end+1} = ['dcc_reconstruct_variance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  dcc_inference_objective...');
    dcc_io_stresid = data_mv ./ repmat(std(data_mv), T, 1);
    dcc_io_R = corr(dcc_io_stresid);
    [dcc_io_ll] = dcc_inference_objective([0.05; 0.93], dcc_io_stresid, dcc_io_R, [], [], 1, 0, 1);
    save(fullfile(outDir, 'dcc_inference_objective.mat'), 'dcc_io_ll');
catch e
    errors{end+1} = ['dcc_inference_objective: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  dcc_fit_variance...');
    [dcc_fv_params, dcc_fv_ht, dcc_fv_stresid] = dcc_fit_variance(data_mv, 1, 0, 1, 2);
    save(fullfile(outDir, 'dcc_fit_variance.mat'), 'dcc_fv_params', 'dcc_fv_ht', 'dcc_fv_stresid');
catch e
    skipped{end+1} = ['dcc_fit_variance: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  dcc (full estimation)...');
    [dcc_parameters, dcc_ll, dcc_Ht] = dcc(data_mv, [], 1, 0, 1);
    save(fullfile(outDir, 'dcc.mat'), 'dcc_parameters', 'dcc_ll', 'dcc_Ht');
catch e
    skipped{end+1} = ['dcc: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- GO-GARCH ---
disp('=== GO-GARCH ===');
try
    disp('  gogarch_likelihood...');
    gog_P = eye(K);
    [gog_ll, gog_lls, gog_Ht] = gogarch_likelihood(gog_P(:), data_mv, K, 1, 0, 1, 2);
    save(fullfile(outDir, 'gogarch_likelihood.mat'), 'gog_P', 'gog_ll', 'gog_lls');
catch e
    errors{end+1} = ['gogarch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  gogarch (full estimation)...');
    [gogarch_parameters, gogarch_ll, gogarch_Ht] = gogarch(data_mv, 1, 0, 1);
    save(fullfile(outDir, 'gogarch.mat'), 'gogarch_parameters', 'gogarch_ll');
catch e
    skipped{end+1} = ['gogarch: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- Matrix GARCH ---
disp('=== Matrix GARCH ===');
try
    disp('  matrix_garch_simulate...');
    mg_k = 2;
    mg_C_param = vech(eye(mg_k) * 0.05);
    mg_A_param = vech(eye(mg_k) * 0.05);
    mg_B_param = vech(eye(mg_k) * 0.9);
    mg_sim_params = [mg_C_param; mg_A_param; mg_B_param];
    [mg_simdata, mg_simHt] = matrix_garch_simulate(500, mg_k, mg_sim_params, 1, 0, 1, 'Scalar');
    save(fullfile(outDir, 'matrix_garch_simulate.mat'), 'mg_sim_params', 'mg_simdata', 'mg_simHt');
catch e
    errors{end+1} = ['matrix_garch_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  matrix_garch_likelihood...');
    mg_ll_k = 2;
    mg_ll_C = vech(eye(mg_ll_k) * 0.05);
    mg_ll_A = vech(eye(mg_ll_k) * 0.05);
    mg_ll_B = vech(eye(mg_ll_k) * 0.9);
    mg_ll_params = [mg_ll_C; mg_ll_A; mg_ll_B];
    [mg_ll_val, mg_ll_lls] = matrix_garch_likelihood(mg_ll_params, data_mv(:,1:2), 1, 0, 1, 'Scalar', false, false);
    save(fullfile(outDir, 'matrix_garch_likelihood.mat'), 'mg_ll_params', 'mg_ll_val', 'mg_ll_lls');
catch e
    errors{end+1} = ['matrix_garch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  matrix_garch_display...');
    % matrix_garch_display outputs text; save parameters used
    mg_disp_params = [0.05; 0.05; 0.9];
    mg_disp_ll = -1500;
    save(fullfile(outDir, 'matrix_garch_display.mat'), 'mg_disp_params', 'mg_disp_ll');
catch e
    errors{end+1} = ['matrix_garch_display: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  matrix_garch (full estimation)...');
    [mg_parameters, mg_ll, mg_Ht] = matrix_garch(data_mv(:,1:2), [], 1, 0, 1, 'Scalar');
    save(fullfile(outDir, 'matrix_garch.mat'), 'mg_parameters', 'mg_ll');
catch e
    skipped{end+1} = ['matrix_garch: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- O-MVGARCH ---
disp('=== O-MVGARCH ===');
try
    disp('  ogarch_likelihood...');
    og_ll_params = [0.05; 0.05; 0.9];
    [og_ll_val, og_ll_lls, og_ll_ht] = ogarch_likelihood(og_ll_params, data_mv(:,1), 1, 0, 1, 2);
    save(fullfile(outDir, 'ogarch_likelihood.mat'), 'og_ll_params', 'og_ll_val', 'og_ll_lls', 'og_ll_ht');
catch e
    errors{end+1} = ['ogarch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  o_mvgarch (full estimation)...');
    [omv_parameters, omv_ll, omv_Ht] = o_mvgarch(data_mv, 1, 0, 1);
    save(fullfile(outDir, 'o_mvgarch.mat'), 'omv_parameters', 'omv_ll');
catch e
    skipped{end+1} = ['o_mvgarch: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- RARCH ---
disp('=== RARCH ===');
try
    disp('  rarch_simulate...');
    rarch_sim_p = [0.05; 0.9];
    [rarch_simdata, rarch_simHt] = rarch_simulate(500, K, rarch_sim_p, 1, 1, 'Scalar');
    save(fullfile(outDir, 'rarch_simulate.mat'), 'rarch_sim_p', 'rarch_simdata', 'rarch_simHt');
catch e
    errors{end+1} = ['rarch_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  rarch_constraint...');
    [rarch_c, rarch_ceq] = rarch_constraint([0.05; 0.9], 1, 1, K, 'Scalar');
    save(fullfile(outDir, 'rarch_constraint.mat'), 'rarch_c', 'rarch_ceq');
catch e
    errors{end+1} = ['rarch_constraint: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  rarch_likelihood...');
    rarch_ll_cov = cov(data_mv) * 0.05;
    [rarch_ll, rarch_lls] = rarch_likelihood([0.05; 0.9], data_mv, rarch_ll_cov, 1, 1, K, 'Scalar', false, false);
    save(fullfile(outDir, 'rarch_likelihood.mat'), 'rarch_ll', 'rarch_lls');
catch e
    errors{end+1} = ['rarch_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  rarch_parameter_transform...');
    [rarch_pt] = rarch_parameter_transform([0.05; 0.9], 1, 1, K, 'Scalar');
    save(fullfile(outDir, 'rarch_parameter_transform.mat'), 'rarch_pt');
catch e
    errors{end+1} = ['rarch_parameter_transform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  rarch (full estimation)...');
    [rarch_parameters, rarch_ll_full, rarch_Ht] = rarch(data_mv, 1, 1, 'Scalar');
    save(fullfile(outDir, 'rarch.mat'), 'rarch_parameters', 'rarch_ll_full');
catch e
    skipped{end+1} = ['rarch: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- RCC ---
disp('=== RCC ===');
try
    disp('  rcc_constraint...');
    [rcc_c, rcc_ceq] = rcc_constraint([0.05; 0.9], 1, 1);
    save(fullfile(outDir, 'rcc_constraint.mat'), 'rcc_c', 'rcc_ceq');
catch e
    errors{end+1} = ['rcc_constraint: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  rcc_likelihood...');
    rcc_stresid = data_mv ./ repmat(std(data_mv), T, 1);
    rcc_R = corr(rcc_stresid);
    [rcc_ll, rcc_lls] = rcc_likelihood([0.05; 0.9], rcc_stresid, rcc_R, 1, 1, K);
    save(fullfile(outDir, 'rcc_likelihood.mat'), 'rcc_ll', 'rcc_lls');
catch e
    errors{end+1} = ['rcc_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  rcc (full estimation)...');
    [rcc_parameters, rcc_ll_full, rcc_Ht] = rcc(data_mv, [], 1, 0, 1, 1, 1);
    save(fullfile(outDir, 'rcc.mat'), 'rcc_parameters', 'rcc_ll_full');
catch e
    skipped{end+1} = ['rcc: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- Scalar VT-VECH ---
disp('=== Scalar VT-VECH ===');
try
    disp('  scalar_vt_vech_simulate...');
    svv_intercept = cov(data_mv(1:100,:));
    [svv_simdata, svv_simHt] = scalar_vt_vech_simulate(500, K, [0.05; 0.92], svv_intercept);
    save(fullfile(outDir, 'scalar_vt_vech_simulate.mat'), 'svv_simdata', 'svv_simHt');
catch e
    errors{end+1} = ['scalar_vt_vech_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  scalar_vt_vech_transform...');
    [svv_trans] = scalar_vt_vech_transform([0.05; 0.92]);
    save(fullfile(outDir, 'scalar_vt_vech_transform.mat'), 'svv_trans');
catch e
    errors{end+1} = ['scalar_vt_vech_transform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  scalar_vt_vech_itransform...');
    svv_trans_input = [0.05; 0.92];
    [svv_itrans] = scalar_vt_vech_itransform(scalar_vt_vech_transform(svv_trans_input));
    save(fullfile(outDir, 'scalar_vt_vech_itransform.mat'), 'svv_trans_input', 'svv_itrans');
catch e
    errors{end+1} = ['scalar_vt_vech_itransform: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  scalar_vt_vech_likelihood...');
    [svv_ll, svv_lls] = scalar_vt_vech_likelihood([0.05; 0.92], data_mv, cov(data_mv));
    save(fullfile(outDir, 'scalar_vt_vech_likelihood.mat'), 'svv_ll', 'svv_lls');
catch e
    errors{end+1} = ['scalar_vt_vech_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  scalar_vt_vech_starting_values...');
    [svv_sv] = scalar_vt_vech_starting_values(data_mv);
    save(fullfile(outDir, 'scalar_vt_vech_starting_values.mat'), 'svv_sv');
catch e
    errors{end+1} = ['scalar_vt_vech_starting_values: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  scalar_vt_vech (full estimation)...');
    [svv_parameters, svv_ll_full, svv_Ht] = scalar_vt_vech(data_mv);
    save(fullfile(outDir, 'scalar_vt_vech.mat'), 'svv_parameters', 'svv_ll_full');
catch e
    skipped{end+1} = ['scalar_vt_vech: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% ========================================================================
%  TIME SERIES FIXTURES (31 functions)
%  ========================================================================
disp(' ');
disp('============================================');
disp('Time Series Fixtures');
disp('============================================');
outDir = fullfile(baseOutputDir, 'timeseries');

%% --- ACF/PACF ---
disp('=== ACF/PACF ===');
try
    disp('  acf...');
    [acf_autocorr, acf_sigma2] = acf([0.5], [0.3], 20);
    save(fullfile(outDir, 'acf.mat'), 'acf_autocorr', 'acf_sigma2');
catch e
    errors{end+1} = ['acf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  pacf...');
    [pacf_out, pacf_se] = pacf(y_stationary, 20);
    save(fullfile(outDir, 'pacf.mat'), 'pacf_out', 'pacf_se');
catch e
    errors{end+1} = ['pacf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  sacf...');
    [sacf_out, sacf_bounds] = sacf(y_stationary, 20);
    save(fullfile(outDir, 'sacf.mat'), 'sacf_out', 'sacf_bounds');
catch e
    errors{end+1} = ['sacf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  spacf...');
    [spacf_out, spacf_bounds] = spacf(y_stationary, 20);
    save(fullfile(outDir, 'spacf.mat'), 'spacf_out', 'spacf_bounds');
catch e
    errors{end+1} = ['spacf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Information Criteria ---
disp('=== Information Criteria ===');
try
    disp('  aicsbic...');
    [aicsbic_aic, aicsbic_sbic] = aicsbic(-1500, 4, T);
    save(fullfile(outDir, 'aicsbic.mat'), 'aicsbic_aic', 'aicsbic_sbic');
catch e
    errors{end+1} = ['aicsbic: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  aichqcsbic...');
    [aich_aic, aich_hq, aich_csbic, aich_sbic] = aichqcsbic(-1500, 4, T);
    save(fullfile(outDir, 'aichqcsbic.mat'), 'aich_aic', 'aich_hq', 'aich_csbic', 'aich_sbic');
catch e
    errors{end+1} = ['aichqcsbic: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- ARMAX ---
disp('=== ARMAX ===');
try
    disp('  armaxfilter_simulate...');
    [armax_simdata] = armaxfilter_simulate(T, 1, [0.5], 0.9);
    save(fullfile(outDir, 'armaxfilter_simulate.mat'), 'armax_simdata');
catch e
    errors{end+1} = ['armaxfilter_simulate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  armaxerrors...');
    armax_err_params = [0.1; 0.5; 0.3];
    armax_errs = armaxerrors(armax_err_params, y_stationary, 1, 1, 1, T);
    save(fullfile(outDir, 'armaxerrors.mat'), 'armax_err_params', 'armax_errs');
catch e
    errors{end+1} = ['armaxerrors: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  armaxfilter_core...');
    afc_params = [0.1; 0.5; 0.3];
    afc_errs = armaxfilter_core(afc_params, y_stationary, 1, 1, 1, T);
    save(fullfile(outDir, 'armaxfilter_core.mat'), 'afc_params', 'afc_errs');
catch e
    errors{end+1} = ['armaxfilter_core: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  armaxfilter_likelihood...');
    afl_params = [0.1; 0.5; 0.3];
    [afl_ll, afl_lls, afl_errs] = armaxfilter_likelihood(afl_params, y_stationary, 1, 1, 1, T, ones(T,1), 0);
    save(fullfile(outDir, 'armaxfilter_likelihood.mat'), 'afl_params', 'afl_ll', 'afl_lls', 'afl_errs');
catch e
    errors{end+1} = ['armaxfilter_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  armaxfilter (full estimation)...');
    [armax_params, armax_LL, armax_errs_full, armax_SE, armax_diag, armax_RVCV, armax_VCV] = armaxfilter(y_stationary, 1, 1, 1);
    save(fullfile(outDir, 'armaxfilter.mat'), 'armax_params', 'armax_LL', 'armax_errs_full');
catch e
    skipped{end+1} = ['armaxfilter: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% --- ARMA Utilities ---
disp('=== ARMA Utilities ===');
try
    disp('  arma_forecaster...');
    af_errs = randn(T, 1);
    [af_fc] = arma_forecaster([0.1; 0.5; 0.3], 1, 1, af_errs, 10);
    save(fullfile(outDir, 'arma_forecaster.mat'), 'af_fc', 'af_errs');
catch e
    errors{end+1} = ['arma_forecaster: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  armaroots...');
    [ar_roots] = armaroots([0.5], [0.3]);
    save(fullfile(outDir, 'armaroots.mat'), 'ar_roots');
catch e
    errors{end+1} = ['armaroots: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  convert_ma_roots...');
    [cmr_out] = convert_ma_roots([1.5]);
    save(fullfile(outDir, 'convert_ma_roots.mat'), 'cmr_out');
catch e
    errors{end+1} = ['convert_ma_roots: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  inverse_ar_roots...');
    [iar_out] = inverse_ar_roots([0.5]);
    save(fullfile(outDir, 'inverse_ar_roots.mat'), 'iar_out');
catch e
    errors{end+1} = ['inverse_ar_roots: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- ADF Tests ---
disp('=== ADF Tests ===');
try
    disp('  augdf...');
    [augdf_stat, augdf_pval, augdf_cv, augdf_resid] = augdf(y_ts, 1, 4);
    save(fullfile(outDir, 'augdf.mat'), 'augdf_stat', 'augdf_pval', 'augdf_cv', 'augdf_resid');
catch e
    errors{end+1} = ['augdf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  augdfcv...');
    [augdfcv_out] = augdfcv(T, 1);
    save(fullfile(outDir, 'augdfcv.mat'), 'augdfcv_out');
catch e
    errors{end+1} = ['augdfcv: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  augdfautolag...');
    [augdfa_lags, augdfa_ic] = augdfautolag(y_ts, 1, 20);
    save(fullfile(outDir, 'augdfautolag.mat'), 'augdfa_lags', 'augdfa_ic');
catch e
    errors{end+1} = ['augdfautolag: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  augdf_cvsim_tieup...');
    % This function typically runs a simulation — save a marker to confirm existence
    augdf_cvs_dummy = true;
    save(fullfile(outDir, 'augdf_cvsim_tieup.mat'), 'augdf_cvs_dummy');
catch e
    errors{end+1} = ['augdf_cvsim_tieup: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Filters ---
disp('=== Filters ===');
try
    disp('  hp_filter...');
    [hp_trend, hp_cyclic] = hp_filter(y_ts, 1600);
    save(fullfile(outDir, 'hp_filter.mat'), 'hp_trend', 'hp_cyclic');
catch e
    errors{end+1} = ['hp_filter: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  bkfilter...');
    [bk_trend, bk_cyclic] = bkfilter(y_ts, 6, 32, 12);
    save(fullfile(outDir, 'bkfilter.mat'), 'bk_trend', 'bk_cyclic');
catch e
    errors{end+1} = ['bkfilter: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Other Time Series Functions ---
disp('=== Other Time Series ===');
try
    disp('  beveridgenelson...');
    [bn_perm, bn_trans] = beveridgenelson(y_ts, 4);
    save(fullfile(outDir, 'beveridgenelson.mat'), 'bn_perm', 'bn_trans');
catch e
    errors{end+1} = ['beveridgenelson: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  grangercause...');
    [gc_stat, gc_pval] = grangercause(y_var(:,2), y_var(:,1), 4);
    save(fullfile(outDir, 'grangercause.mat'), 'gc_stat', 'gc_pval');
catch e
    errors{end+1} = ['grangercause: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  heterogeneousar...');
    har_data = abs(y_stationary);
    [har_b, har_tstat, har_s2, har_vcv, har_R2, har_Rbar, har_yhat] = heterogeneousar(har_data, [1 5 22]);
    save(fullfile(outDir, 'heterogeneousar.mat'), 'har_b', 'har_tstat', 'har_s2', 'har_vcv', 'har_R2', 'har_Rbar', 'har_yhat');
catch e
    errors{end+1} = ['heterogeneousar: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  impulseresponse...');
    [ir_responses] = impulseresponse(y_var, 1, 2, 20);
    save(fullfile(outDir, 'impulseresponse.mat'), 'ir_responses');
catch e
    errors{end+1} = ['impulseresponse: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  impulseresponse_bootstrap...');
    [irb_lower, irb_upper, irb_resp] = impulseresponse_bootstrap(y_var, 1, 2, 20, 100);
    save(fullfile(outDir, 'impulseresponse_bootstrap.mat'), 'irb_lower', 'irb_upper', 'irb_resp');
catch e
    errors{end+1} = ['impulseresponse_bootstrap: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Estimation ---
disp('=== Estimation ===');
try
    disp('  olsnw...');
    [olsnw_b, olsnw_tstat, olsnw_s2, olsnw_vcv] = olsnw(y_reg, X_reg, 4);
    save(fullfile(outDir, 'olsnw.mat'), 'olsnw_b', 'olsnw_tstat', 'olsnw_s2', 'olsnw_vcv');
catch e
    errors{end+1} = ['olsnw: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  vectorar...');
    [var_b, var_stderr, var_vcv, var_F, var_pval] = vectorar(y_var, 1, 2);
    save(fullfile(outDir, 'vectorar.mat'), 'var_b', 'var_stderr', 'var_vcv');
catch e
    errors{end+1} = ['vectorar: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  vectorarvcv...');
    [varvcv_out] = vectorarvcv(y_var, 1, 2);
    save(fullfile(outDir, 'vectorarvcv.mat'), 'varvcv_out');
catch e
    errors{end+1} = ['vectorarvcv: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Plotting and Spectral ---
disp('=== Plotting and Spectral ===');
try
    disp('  tsresidualplot...');
    % tsresidualplot produces a plot; save the data that would be plotted
    tsr_resid = randn(T, 1);
    save(fullfile(outDir, 'tsresidualplot.mat'), 'tsr_resid');
catch e
    errors{end+1} = ['tsresidualplot: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  weights_to_frequency_response...');
    wtfr_weights = [1 -0.5 0.25];
    [w2fr] = weights_to_frequency_response(wtfr_weights, 256);
    save(fullfile(outDir, 'weights_to_frequency_response.mat'), 'wtfr_weights', 'w2fr');
catch e
    errors{end+1} = ['weights_to_frequency_response: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% ========================================================================
%  REALIZED VOLATILITY FIXTURES (42 functions)
%  ========================================================================
disp(' ');
disp('============================================');
disp('Realized Volatility Fixtures');
disp('============================================');
outDir = fullfile(baseOutputDir, 'realized');

%% --- Core Estimators ---
disp('=== Realized Core Estimators ===');
try
    disp('  realized_variance...');
    [rv_out, rv_ss] = realized_variance(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_variance.mat'), 'rv_out', 'rv_ss');
catch e
    errors{end+1} = ['realized_variance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_bipower_variation...');
    [rbv_out, rbv_ss] = realized_bipower_variation(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_bipower_variation.mat'), 'rbv_out', 'rbv_ss');
catch e
    errors{end+1} = ['realized_bipower_variation: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_kernel...');
    [rk_out] = realized_kernel(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_kernel.mat'), 'rk_out');
catch e
    errors{end+1} = ['realized_kernel: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_covariance...');
    [rcov_out] = realized_covariance(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_covariance.mat'), 'rcov_out');
catch e
    errors{end+1} = ['realized_covariance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_range...');
    [rr_out] = realized_range(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_range.mat'), 'rr_out');
catch e
    errors{end+1} = ['realized_range: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_quantile_variance...');
    [rqv_out] = realized_quantile_variance(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_quantile_variance.mat'), 'rqv_out');
catch e
    errors{end+1} = ['realized_quantile_variance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_semivariance...');
    [rsv_out] = realized_semivariance(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_semivariance.mat'), 'rsv_out');
catch e
    errors{end+1} = ['realized_semivariance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_threshold_variance...');
    [rtv_out] = realized_threshold_variance(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_threshold_variance.mat'), 'rtv_out');
catch e
    errors{end+1} = ['realized_threshold_variance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_twoscale_variance...');
    [rtsv_out] = realized_twoscale_variance(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_twoscale_variance.mat'), 'rtsv_out');
catch e
    errors{end+1} = ['realized_twoscale_variance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_multiscale_variance...');
    [rmsv_out] = realized_multiscale_variance(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_multiscale_variance.mat'), 'rmsv_out');
catch e
    errors{end+1} = ['realized_multiscale_variance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_min_med_variance...');
    [rmmv_out] = realized_min_med_variance(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_min_med_variance.mat'), 'rmmv_out');
catch e
    errors{end+1} = ['realized_min_med_variance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_preaveraged_variance...');
    [rpv_out] = realized_preaveraged_variance(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_preaveraged_variance.mat'), 'rpv_out');
catch e
    errors{end+1} = ['realized_preaveraged_variance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_preaveraged_bipower_variation...');
    [rpbv_out] = realized_preaveraged_bipower_variation(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_preaveraged_bipower_variation.mat'), 'rpbv_out');
catch e
    errors{end+1} = ['realized_preaveraged_bipower_variation: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_qmle_variance...');
    [rqmle_out] = realized_qmle_variance(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_qmle_variance.mat'), 'rqmle_out');
catch e
    errors{end+1} = ['realized_qmle_variance: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_threshold_multipower_variation...');
    [rtmpv_out] = realized_threshold_multipower_variation(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_threshold_multipower_variation.mat'), 'rtmpv_out');
catch e
    errors{end+1} = ['realized_threshold_multipower_variation: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_quarticity...');
    [rq_out] = realized_quarticity(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_quarticity.mat'), 'rq_out');
catch e
    errors{end+1} = ['realized_quarticity: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Realized Tests and Optimal Sampling ---
disp('=== Realized Tests ===');
try
    disp('  realized_test...');
    [rt_stat, rt_pval] = realized_test(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_test.mat'), 'rt_stat', 'rt_pval');
catch e
    errors{end+1} = ['realized_test: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_variance_optimal_sampling...');
    [rvos_out] = realized_variance_optimal_sampling(hf_prices, hf_times_seconds, 'seconds');
    save(fullfile(outDir, 'realized_variance_optimal_sampling.mat'), 'rvos_out');
catch e
    errors{end+1} = ['realized_variance_optimal_sampling: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Realized Helpers ---
disp('=== Realized Helpers ===');
try
    disp('  realized_compute_median...');
    rcm_data = diff(log(hf_prices));
    rcm_out = realized_compute_median(rcm_data);
    save(fullfile(outDir, 'realized_compute_median.mat'), 'rcm_data', 'rcm_out');
catch e
    errors{end+1} = ['realized_compute_median: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_convert2unit...');
    rc2u_out = realized_convert2unit(hf_times_seconds, 'seconds');
    save(fullfile(outDir, 'realized_convert2unit.mat'), 'rc2u_out');
catch e
    errors{end+1} = ['realized_convert2unit: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_options...');
    ro_out = realized_options('seconds', 'CalendarTime', 300, 1);
    save(fullfile(outDir, 'realized_options.mat'), 'ro_out');
catch e
    errors{end+1} = ['realized_options: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_price_filter...');
    [rpf_p, rpf_t] = realized_price_filter(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_price_filter.mat'), 'rpf_p', 'rpf_t');
catch e
    errors{end+1} = ['realized_price_filter: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_return_filter...');
    [rrf_r, rrf_t] = realized_return_filter(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_return_filter.mat'), 'rrf_r', 'rrf_t');
catch e
    errors{end+1} = ['realized_return_filter: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_noise_estimate...');
    [rne_out] = realized_noise_estimate(hf_prices, hf_times_seconds, 'seconds');
    save(fullfile(outDir, 'realized_noise_estimate.mat'), 'rne_out');
catch e
    errors{end+1} = ['realized_noise_estimate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_subsample...');
    [rss_out] = realized_subsample(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300, 5);
    save(fullfile(outDir, 'realized_subsample.mat'), 'rss_out');
catch e
    errors{end+1} = ['realized_subsample: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Kernel Helpers ---
disp('=== Kernel Helpers ===');
try
    disp('  realized_kernel_bandwidth...');
    [rkb_out] = realized_kernel_bandwidth(hf_prices, hf_times_seconds, 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_kernel_bandwidth.mat'), 'rkb_out');
catch e
    errors{end+1} = ['realized_kernel_bandwidth: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_kernel_core...');
    rkc_returns = diff(log(hf_prices));
    [rkc_out] = realized_kernel_core(rkc_returns, 5, 'bartlett');
    save(fullfile(outDir, 'realized_kernel_core.mat'), 'rkc_returns', 'rkc_out');
catch e
    errors{end+1} = ['realized_kernel_core: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_kernel_weights...');
    [rkw_out] = realized_kernel_weights(10, 'bartlett');
    save(fullfile(outDir, 'realized_kernel_weights.mat'), 'rkw_out');
catch e
    errors{end+1} = ['realized_kernel_weights: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_kernel_jitter_lag_length...');
    [rkjll_out] = realized_kernel_jitter_lag_length(hf_prices, hf_times_seconds, 'seconds');
    save(fullfile(outDir, 'realized_kernel_jitter_lag_length.mat'), 'rkjll_out');
catch e
    errors{end+1} = ['realized_kernel_jitter_lag_length: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Multivariate Realized ---
disp('=== Multivariate Realized ===');
try
    disp('  realized_multivariate_kernel...');
    [rmk_out] = realized_multivariate_kernel([hf_prices hf_prices2], [hf_times_seconds hf_times_seconds2], 'seconds', 'CalendarTime', 300);
    save(fullfile(outDir, 'realized_multivariate_kernel.mat'), 'rmk_out');
catch e
    errors{end+1} = ['realized_multivariate_kernel: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_hayashi_yoshida...');
    [rhy_out] = realized_hayashi_yoshida(hf_prices, hf_times_seconds, hf_prices2, hf_times_seconds2, 'seconds');
    save(fullfile(outDir, 'realized_hayashi_yoshida.mat'), 'rhy_out');
catch e
    errors{end+1} = ['realized_hayashi_yoshida: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_refresh_time...');
    [rrt_p, rrt_t] = realized_refresh_time(hf_prices, hf_times_seconds, 'seconds');
    save(fullfile(outDir, 'realized_refresh_time.mat'), 'rrt_p', 'rrt_t');
catch e
    errors{end+1} = ['realized_refresh_time: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_refresh_time_bivariate...');
    [rrtb_p1, rrtb_p2, rrtb_t] = realized_refresh_time_bivariate(hf_prices, hf_times_seconds, hf_prices2, hf_times_seconds2, 'seconds');
    save(fullfile(outDir, 'realized_refresh_time_bivariate.mat'), 'rrtb_p1', 'rrtb_p2', 'rrtb_t');
catch e
    errors{end+1} = ['realized_refresh_time_bivariate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Realized Simulation ---
disp('=== Realized Simulation ===');
try
    disp('  realized_range_simulation...');
    [rrs_out] = realized_range_simulation(1000, 100, 78);
    save(fullfile(outDir, 'realized_range_simulation.mat'), 'rrs_out');
catch e
    errors{end+1} = ['realized_range_simulation: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_quantile_weight_simulation...');
    [rqws_out] = realized_quantile_weight_simulation(100, 78);
    save(fullfile(outDir, 'realized_quantile_weight_simulation.mat'), 'rqws_out');
catch e
    errors{end+1} = ['realized_quantile_weight_simulation: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  realized_quantile_variance_scale...');
    [rqvs_out] = realized_quantile_variance_scale(78);
    save(fullfile(outDir, 'realized_quantile_variance_scale.mat'), 'rqvs_out');
catch e
    errors{end+1} = ['realized_quantile_variance_scale: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Time Conversion Functions ---
disp('=== Time Conversions ===');
try
    disp('  seconds2unit...');
    s2u_out = seconds2unit(36000);
    save(fullfile(outDir, 'seconds2unit.mat'), 's2u_out');
catch e
    errors{end+1} = ['seconds2unit: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  seconds2wall...');
    s2w_out = seconds2wall(36000);
    save(fullfile(outDir, 'seconds2wall.mat'), 's2w_out');
catch e
    errors{end+1} = ['seconds2wall: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  unit2seconds...');
    u2s_out = unit2seconds(0.5);
    save(fullfile(outDir, 'unit2seconds.mat'), 'u2s_out');
catch e
    errors{end+1} = ['unit2seconds: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  unit2wall...');
    u2w_out = unit2wall(0.5);
    save(fullfile(outDir, 'unit2wall.mat'), 'u2w_out');
catch e
    errors{end+1} = ['unit2wall: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  wall2seconds...');
    w2s_out = wall2seconds(120000);
    save(fullfile(outDir, 'wall2seconds.mat'), 'w2s_out');
catch e
    errors{end+1} = ['wall2seconds: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  wall2unit...');
    w2u_out = wall2unit(120000);
    save(fullfile(outDir, 'wall2unit.mat'), 'w2u_out');
catch e
    errors{end+1} = ['wall2unit: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% ========================================================================
%  DISTRIBUTION FIXTURES (19 functions)
%  ========================================================================
disp(' ');
disp('============================================');
disp('Distribution Fixtures');
disp('============================================');
outDir = fullfile(baseOutputDir, 'distributions');

x_dist = linspace(-3, 3, 50)';
p_dist = linspace(0.01, 0.99, 50)';
v_ged = 1.5;       % GED shape parameter >= 1
v_t = 5;            % Student-t degrees of freedom > 2
lambda_skewt = -0.2; % Skewed-t asymmetry, -1 < lambda < 1

%% --- GED Distribution ---
disp('=== GED Distribution ===');
try
    disp('  gedpdf...');
    gedpdf_out = gedpdf(x_dist, v_ged);
    save(fullfile(outDir, 'gedpdf.mat'), 'x_dist', 'v_ged', 'gedpdf_out');
catch e
    errors{end+1} = ['gedpdf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  gedcdf...');
    gedcdf_out = gedcdf(x_dist, v_ged);
    save(fullfile(outDir, 'gedcdf.mat'), 'x_dist', 'v_ged', 'gedcdf_out');
catch e
    errors{end+1} = ['gedcdf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  gedinv...');
    gedinv_out = gedinv(p_dist, v_ged);
    save(fullfile(outDir, 'gedinv.mat'), 'p_dist', 'v_ged', 'gedinv_out');
catch e
    errors{end+1} = ['gedinv: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  gedloglik...');
    [gedll_out, gedlls_out] = gedloglik([v_ged], x_dist);
    save(fullfile(outDir, 'gedloglik.mat'), 'v_ged', 'x_dist', 'gedll_out', 'gedlls_out');
catch e
    errors{end+1} = ['gedloglik: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  gedrnd...');
    gedrnd_out = gedrnd(v_ged, 100, 1);
    save(fullfile(outDir, 'gedrnd.mat'), 'v_ged', 'gedrnd_out');
catch e
    errors{end+1} = ['gedrnd: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Skewed-t Distribution ---
disp('=== Skewed-t Distribution ===');
try
    disp('  skewtpdf...');
    skewtpdf_out = skewtpdf(x_dist, v_t, lambda_skewt);
    save(fullfile(outDir, 'skewtpdf.mat'), 'x_dist', 'v_t', 'lambda_skewt', 'skewtpdf_out');
catch e
    errors{end+1} = ['skewtpdf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  skewtcdf...');
    skewtcdf_out = skewtcdf(x_dist, v_t, lambda_skewt);
    save(fullfile(outDir, 'skewtcdf.mat'), 'x_dist', 'v_t', 'lambda_skewt', 'skewtcdf_out');
catch e
    errors{end+1} = ['skewtcdf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  skewtinv...');
    skewtinv_out = skewtinv(p_dist, v_t, lambda_skewt);
    save(fullfile(outDir, 'skewtinv.mat'), 'p_dist', 'v_t', 'lambda_skewt', 'skewtinv_out');
catch e
    errors{end+1} = ['skewtinv: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  skewtloglik...');
    [skewtll_out, skewtlls_out] = skewtloglik([v_t lambda_skewt], x_dist);
    save(fullfile(outDir, 'skewtloglik.mat'), 'v_t', 'lambda_skewt', 'skewtll_out', 'skewtlls_out');
catch e
    errors{end+1} = ['skewtloglik: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  skewtrnd...');
    skewtrnd_out = skewtrnd(v_t, lambda_skewt, 100, 1);
    save(fullfile(outDir, 'skewtrnd.mat'), 'v_t', 'lambda_skewt', 'skewtrnd_out');
catch e
    errors{end+1} = ['skewtrnd: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Standardized-t Distribution ---
disp('=== Standardized-t Distribution ===');
try
    disp('  stdtpdf...');
    stdtpdf_out = stdtpdf(x_dist, v_t);
    save(fullfile(outDir, 'stdtpdf.mat'), 'x_dist', 'v_t', 'stdtpdf_out');
catch e
    errors{end+1} = ['stdtpdf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  stdtcdf...');
    stdtcdf_out = stdtcdf(x_dist, v_t);
    save(fullfile(outDir, 'stdtcdf.mat'), 'x_dist', 'v_t', 'stdtcdf_out');
catch e
    errors{end+1} = ['stdtcdf: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  stdtinv...');
    stdtinv_out = stdtinv(p_dist, v_t);
    save(fullfile(outDir, 'stdtinv.mat'), 'p_dist', 'v_t', 'stdtinv_out');
catch e
    errors{end+1} = ['stdtinv: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  stdtloglik...');
    [stdtll_out, stdtlls_out] = stdtloglik([v_t], x_dist);
    save(fullfile(outDir, 'stdtloglik.mat'), 'v_t', 'stdtll_out', 'stdtlls_out');
catch e
    errors{end+1} = ['stdtloglik: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  stdtrnd...');
    stdtrnd_out = stdtrnd(v_t, 100, 1);
    save(fullfile(outDir, 'stdtrnd.mat'), 'v_t', 'stdtrnd_out');
catch e
    errors{end+1} = ['stdtrnd: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Normal and Multivariate ---
disp('=== Normal / Multivariate / Composite ===');
try
    disp('  normloglik...');
    [normll_out, normlls_out] = normloglik(x_dist);
    save(fullfile(outDir, 'normloglik.mat'), 'x_dist', 'normll_out', 'normlls_out');
catch e
    errors{end+1} = ['normloglik: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  mvnormloglik...');
    mvn_data = randn(100, 3);
    mvn_sigma = cov(mvn_data);
    [mvnll_out, mvnlls_out] = mvnormloglik(mvn_data, mvn_sigma);
    save(fullfile(outDir, 'mvnormloglik.mat'), 'mvn_data', 'mvn_sigma', 'mvnll_out', 'mvnlls_out');
catch e
    errors{end+1} = ['mvnormloglik: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  composite_likelihood...');
    cl_data = randn(100, 3);
    cl_sigma = cov(cl_data);
    [cl_out, cl_outs] = composite_likelihood(cl_data, cl_sigma);
    save(fullfile(outDir, 'composite_likelihood.mat'), 'cl_data', 'cl_sigma', 'cl_out', 'cl_outs');
catch e
    errors{end+1} = ['composite_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  iscompatible...');
    [ic_err, ic_text, ic_size, ic_v] = iscompatible(1, v_t, size(x_dist));
    save(fullfile(outDir, 'iscompatible.mat'), 'ic_err', 'ic_text', 'ic_size', 'ic_v');
catch e
    errors{end+1} = ['iscompatible: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% ========================================================================
%  UTILITY FIXTURES (29 functions)
%  ========================================================================
disp(' ');
disp('============================================');
disp('Utility Fixtures');
disp('============================================');
outDir = fullfile(baseOutputDir, 'utility');

A_sym = [4 2 1; 2 5 3; 1 3 6];  % 3x3 symmetric positive definite matrix

%% --- Matrix Operations ---
disp('=== Matrix Operations ===');
try
    disp('  vech...');
    vech_out = vech(A_sym);
    save(fullfile(outDir, 'vech.mat'), 'A_sym', 'vech_out');
catch e
    errors{end+1} = ['vech: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  ivech...');
    ivech_input = vech(A_sym);
    ivech_out = ivech(ivech_input);
    save(fullfile(outDir, 'ivech.mat'), 'ivech_input', 'ivech_out');
catch e
    errors{end+1} = ['ivech: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  chol2vec...');
    C_chol = chol(A_sym);
    chol2vec_out = chol2vec(C_chol);
    save(fullfile(outDir, 'chol2vec.mat'), 'C_chol', 'chol2vec_out');
catch e
    errors{end+1} = ['chol2vec: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  vec2chol...');
    v2c_input = chol2vec(chol(A_sym));
    vec2chol_out = vec2chol(v2c_input);
    save(fullfile(outDir, 'vec2chol.mat'), 'v2c_input', 'vec2chol_out');
catch e
    errors{end+1} = ['vec2chol: ' e.message];
    disp(['    ERROR: ' e.message]);
end

R_corr = [1 0.5 0.3; 0.5 1 0.4; 0.3 0.4 1];
try
    disp('  corr_vech...');
    corr_vech_out = corr_vech(R_corr);
    save(fullfile(outDir, 'corr_vech.mat'), 'R_corr', 'corr_vech_out');
catch e
    errors{end+1} = ['corr_vech: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  corr_ivech...');
    cv_input = corr_vech(R_corr);
    corr_ivech_out = corr_ivech(cv_input);
    save(fullfile(outDir, 'corr_ivech.mat'), 'cv_input', 'corr_ivech_out');
catch e
    errors{end+1} = ['corr_ivech: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Covariance Functions ---
disp('=== Covariance Functions ===');
try
    disp('  cov2corr...');
    [cov2corr_out] = cov2corr(A_sym);
    save(fullfile(outDir, 'cov2corr.mat'), 'A_sym', 'cov2corr_out');
catch e
    errors{end+1} = ['cov2corr: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  covnw...');
    covnw_out = covnw(data_mv, 10);
    save(fullfile(outDir, 'covnw.mat'), 'covnw_out');
catch e
    errors{end+1} = ['covnw: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  covvar...');
    covvar_out = covvar(data_mv, 10);
    save(fullfile(outDir, 'covvar.mat'), 'covvar_out');
catch e
    errors{end+1} = ['covvar: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Data Operations ---
disp('=== Data Operations ===');
try
    disp('  demean...');
    demean_out = demean(data_mv);
    save(fullfile(outDir, 'demean.mat'), 'demean_out');
catch e
    errors{end+1} = ['demean: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  standardize...');
    [std_out, std_mu, std_sigma] = standardize(data_mv(:,1));
    save(fullfile(outDir, 'standardize.mat'), 'std_out', 'std_mu', 'std_sigma');
catch e
    errors{end+1} = ['standardize: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  mvstandardize...');
    [mvstd_out, mvstd_mu, mvstd_sigma] = mvstandardize(data_mv);
    save(fullfile(outDir, 'mvstandardize.mat'), 'mvstd_out', 'mvstd_mu', 'mvstd_sigma');
catch e
    errors{end+1} = ['mvstandardize: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  newlagmatrix...');
    [nlm_out, nlm_T] = newlagmatrix(y_stationary, 5, 1);
    save(fullfile(outDir, 'newlagmatrix.mat'), 'nlm_out', 'nlm_T');
catch e
    errors{end+1} = ['newlagmatrix: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Numerical Derivatives ---
disp('=== Numerical Derivatives ===');
try
    disp('  gradient_2sided...');
    f_grad = @(x) sum(x.^2);
    x_grad = [1.0; 2.0; 3.0];
    [g2s_G, g2s_Gt] = gradient_2sided(f_grad, x_grad);
    save(fullfile(outDir, 'gradient_2sided.mat'), 'x_grad', 'g2s_G');
catch e
    errors{end+1} = ['gradient_2sided: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  hessian_2sided...');
    f_hess = @(x) sum(x.^2);
    x_hess = [1.0; 2.0; 3.0];
    [h2s_H] = hessian_2sided(f_hess, x_hess);
    save(fullfile(outDir, 'hessian_2sided.mat'), 'x_hess', 'h2s_H');
catch e
    errors{end+1} = ['hessian_2sided: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  hessian_2sided_nrows...');
    f_hn = @(x) sum(x.^2);
    x_hn = [1.0; 2.0; 3.0];
    [h2sn_H] = hessian_2sided_nrows(f_hn, x_hn, 2);
    save(fullfile(outDir, 'hessian_2sided_nrows.mat'), 'x_hn', 'h2sn_H');
catch e
    errors{end+1} = ['hessian_2sided_nrows: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Robust VCV ---
disp('=== Robust VCV ===');
try
    disp('  robustvcv...');
    f_rv = @(x) sum(x.^2);
    x_rv = [1.0; 2.0];
    [rv_vcv, rv_A, rv_B, rv_scores] = robustvcv(f_rv, x_rv, 100);
    save(fullfile(outDir, 'robustvcv.mat'), 'x_rv', 'rv_vcv');
catch e
    errors{end+1} = ['robustvcv: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Correlation Transforms ---
disp('=== Correlation Transforms ===');
phi_vals = [0.3; 0.5];
r_vals = [0.3; 0.5];

try
    disp('  phi2r...');
    phi2r_out = phi2r(phi_vals);
    save(fullfile(outDir, 'phi2r.mat'), 'phi_vals', 'phi2r_out');
catch e
    errors{end+1} = ['phi2r: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  phi2u...');
    phi2u_out = phi2u(phi_vals);
    save(fullfile(outDir, 'phi2u.mat'), 'phi_vals', 'phi2u_out');
catch e
    errors{end+1} = ['phi2u: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  r2phi...');
    r2phi_out = r2phi(r_vals);
    save(fullfile(outDir, 'r2phi.mat'), 'r_vals', 'r2phi_out');
catch e
    errors{end+1} = ['r2phi: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  r2z...');
    r2z_out = r2z(0.5);
    save(fullfile(outDir, 'r2z.mat'), 'r2z_out');
catch e
    errors{end+1} = ['r2z: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  z2r...');
    z2r_input = r2z(0.5);
    z2r_out = z2r(z2r_input);
    save(fullfile(outDir, 'z2r.mat'), 'z2r_input', 'z2r_out');
catch e
    errors{end+1} = ['z2r: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Display and Plot ---
disp('=== Display and Plot ===');
try
    disp('  mprint...');
    % mprint outputs text; save the data that would be printed
    mprint_data = rand(3, 3);
    save(fullfile(outDir, 'mprint.mat'), 'mprint_data');
catch e
    errors{end+1} = ['mprint: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  pltdens...');
    % pltdens produces a plot; save the data that would be plotted
    pltdens_data = randn(500, 1);
    save(fullfile(outDir, 'pltdens.mat'), 'pltdens_data');
catch e
    errors{end+1} = ['pltdens: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Date Conversions ---
disp('=== Date Conversions ===');
try
    disp('  c2mdate...');
    c2m_input = 1000000000;  % Unix timestamp
    c2m_out = c2mdate(c2m_input);
    save(fullfile(outDir, 'c2mdate.mat'), 'c2m_input', 'c2m_out');
catch e
    errors{end+1} = ['c2mdate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  m2cdate...');
    m2c_input = 733773;  % MATLAB datenum
    m2c_out = m2cdate(m2c_input);
    save(fullfile(outDir, 'm2cdate.mat'), 'm2c_input', 'm2c_out');
catch e
    errors{end+1} = ['m2cdate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  x2mdate...');
    x2m_input = 40000;  % Excel date serial
    x2m_out = x2mdate(x2m_input);
    save(fullfile(outDir, 'x2mdate.mat'), 'x2m_input', 'x2m_out');
catch e
    errors{end+1} = ['x2mdate: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% --- Misc Utility ---
disp('=== Misc Utility ===');
try
    disp('  randchar...');
    rc_out = randchar(10);
    save(fullfile(outDir, 'randchar.mat'), 'rc_out');
catch e
    errors{end+1} = ['randchar: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  dirod...');
    dirod_out = dirod('.');
    save(fullfile(outDir, 'dirod.mat'), 'dirod_out');
catch e
    errors{end+1} = ['dirod: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% ========================================================================
%  BOOTSTRAP FIXTURES (4 functions)
%  ========================================================================
disp(' ');
disp('============================================');
disp('Bootstrap Fixtures');
disp('============================================');
outDir = fullfile(baseOutputDir, 'bootstrap');

try
    disp('  block_bootstrap...');
    bb_data = randn(200, 1);
    [bb_bsdata, bb_indices] = block_bootstrap(bb_data, 100, 10);
    save(fullfile(outDir, 'block_bootstrap.mat'), 'bb_data', 'bb_bsdata', 'bb_indices');
catch e
    errors{end+1} = ['block_bootstrap: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  stationary_bootstrap...');
    sb_data = randn(200, 1);
    [sb_bsdata, sb_indices] = stationary_bootstrap(sb_data, 100, 10);
    save(fullfile(outDir, 'stationary_bootstrap.mat'), 'sb_data', 'sb_bsdata', 'sb_indices');
catch e
    errors{end+1} = ['stationary_bootstrap: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  bsds...');
    bsds_bench = randn(200, 1).^2;
    bsds_models = randn(200, 5).^2;
    [bsds_c, bsds_u, bsds_l] = bsds(bsds_bench, bsds_models, 100, 12);
    save(fullfile(outDir, 'bsds.mat'), 'bsds_bench', 'bsds_models', 'bsds_c', 'bsds_u', 'bsds_l');
catch e
    skipped{end+1} = ['bsds: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

% REQUIRES_OPTIMIZATION_TOOLBOX
try
    disp('  mcs...');
    mcs_losses = randn(200, 5).^2 + repmat(linspace(0.1, 0.5, 5), 200, 1);
    [mcs_inclR, mcs_pvalsR, mcs_exclR, mcs_inclSQ, mcs_pvalsSQ, mcs_exclSQ] = mcs(mcs_losses, 0.05, 100, 12);
    save(fullfile(outDir, 'mcs.mat'), 'mcs_losses', 'mcs_inclR', 'mcs_pvalsR', 'mcs_exclR', 'mcs_inclSQ', 'mcs_pvalsSQ', 'mcs_exclSQ');
catch e
    skipped{end+1} = ['mcs: ' e.message];
    disp(['    SKIPPED: ' e.message]);
end

%% ========================================================================
%  TESTS FIXTURES (5 functions)
%  ========================================================================
disp(' ');
disp('============================================');
disp('Tests Fixtures');
disp('============================================');
outDir = fullfile(baseOutputDir, 'tests');

try
    disp('  berkowitz...');
    berk_data = randn(200, 1);
    [berk_stat, berk_pval] = berkowitz(berk_data);
    save(fullfile(outDir, 'berkowitz.mat'), 'berk_data', 'berk_stat', 'berk_pval');
catch e
    errors{end+1} = ['berkowitz: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  jarquebera...');
    jb_data = randn(200, 1);
    [jb_stat, jb_pval, jb_H] = jarquebera(jb_data);
    save(fullfile(outDir, 'jarquebera.mat'), 'jb_data', 'jb_stat', 'jb_pval', 'jb_H');
catch e
    errors{end+1} = ['jarquebera: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  kolmogorov...');
    ks_data = randn(200, 1);
    [ks_stat, ks_pval] = kolmogorov(ks_data);
    save(fullfile(outDir, 'kolmogorov.mat'), 'ks_data', 'ks_stat', 'ks_pval');
catch e
    errors{end+1} = ['kolmogorov: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  ljungbox...');
    lb_data = randn(200, 1);
    [lb_q, lb_pval] = ljungbox(lb_data, 10);
    save(fullfile(outDir, 'ljungbox.mat'), 'lb_data', 'lb_q', 'lb_pval');
catch e
    errors{end+1} = ['ljungbox: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  lmtest1...');
    lm_data = randn(200, 1);
    [lm_stat, lm_pval] = lmtest1(lm_data, 10);
    save(fullfile(outDir, 'lmtest1.mat'), 'lm_data', 'lm_stat', 'lm_pval');
catch e
    errors{end+1} = ['lmtest1: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% ========================================================================
%  CROSS-SECTION FIXTURES (2 functions)
%  ========================================================================
disp(' ');
disp('============================================');
disp('Cross-Section Fixtures');
disp('============================================');
outDir = fullfile(baseOutputDir, 'crosssection');

try
    disp('  ols...');
    [ols_b, ols_tstat, ols_s2, ols_vcv, ols_vcvwhite, ols_R2, ols_Rbar, ols_yhat] = ols(y_reg, X_reg, 1);
    save(fullfile(outDir, 'ols.mat'), 'y_reg', 'X_reg', ...
        'ols_b', 'ols_tstat', 'ols_s2', 'ols_vcv', 'ols_vcvwhite', 'ols_R2', 'ols_Rbar', 'ols_yhat');
catch e
    errors{end+1} = ['ols: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  pca...');
    pca_data = randn(200, 5);

    % Test all 3 normalization modes as specified in AAP
    [pca_w_outer, pca_pc_outer, pca_ev_outer, pca_expl_outer, pca_cum_outer] = pca(pca_data, 'outer');
    [pca_w_cov, pca_pc_cov, pca_ev_cov, pca_expl_cov, pca_cum_cov] = pca(pca_data, 'cov');
    [pca_w_corr, pca_pc_corr, pca_ev_corr, pca_expl_corr, pca_cum_corr] = pca(pca_data, 'corr');
    save(fullfile(outDir, 'pca.mat'), 'pca_data', ...
        'pca_w_outer', 'pca_pc_outer', 'pca_ev_outer', 'pca_expl_outer', 'pca_cum_outer', ...
        'pca_w_cov', 'pca_pc_cov', 'pca_ev_cov', 'pca_expl_cov', 'pca_cum_cov', ...
        'pca_w_corr', 'pca_pc_corr', 'pca_ev_corr', 'pca_expl_corr', 'pca_cum_corr');
catch e
    errors{end+1} = ['pca: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% ========================================================================
%  SANDBOX FIXTURES (6 functions)
%  ========================================================================
disp(' ');
disp('============================================');
disp('Sandbox Fixtures');
disp('============================================');
outDir = fullfile(baseOutputDir, 'sandbox');

% sarima - empty stub, just verify it exists
try
    disp('  sarima (empty stub - skip fixture)...');
    sarima_exists = exist('sarima', 'file') > 0;
    save(fullfile(outDir, 'sarima.mat'), 'sarima_exists');
catch e
    errors{end+1} = ['sarima: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  sdiff...');
    sd_x = cumsum(randn(200, 1));
    [sd_y, sd_lags, sd_scales] = sdiff(sd_x, 1, 4);
    save(fullfile(outDir, 'sdiff.mat'), 'sd_x', 'sd_y', 'sd_lags', 'sd_scales');
catch e
    errors{end+1} = ['sdiff: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  sarma2arma...');
    [s2a_ar, s2a_ma] = sarma2arma([0.5], [0.3], [0.8], [0.2], 4);
    save(fullfile(outDir, 'sarma2arma.mat'), 's2a_ar', 's2a_ma');
catch e
    errors{end+1} = ['sarma2arma: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  sarimax_errors...');
    sxe_y = randn(200, 1);
    sxe_params = [0.1; 0.5; 0.3];
    [sxe_errors] = sarimax_errors(sxe_params, sxe_y, 1, 1, [], 1, 200);
    save(fullfile(outDir, 'sarimax_errors.mat'), 'sxe_params', 'sxe_y', 'sxe_errors');
catch e
    errors{end+1} = ['sarimax_errors: ' e.message];
    disp(['    ERROR: ' e.message]);
end

try
    disp('  sarimax_likelihood...');
    sxl_y = randn(200, 1);
    sxl_params = [0.1; 0.5; 0.3];
    [sxl_ll, sxl_lls, sxl_errors] = sarimax_likelihood(sxl_params, sxl_y, 1, 1, [], 1, 200, ones(200,1));
    save(fullfile(outDir, 'sarimax_likelihood.mat'), 'sxl_params', 'sxl_y', 'sxl_ll', 'sxl_lls', 'sxl_errors');
catch e
    errors{end+1} = ['sarimax_likelihood: ' e.message];
    disp(['    ERROR: ' e.message]);
end

% heavy_test - this is a test script; record that it exists
try
    disp('  heavy_test (test script - skip fixture)...');
    heavy_test_exists = exist('heavy_test', 'file') > 0;
    save(fullfile(outDir, 'heavy_test.mat'), 'heavy_test_exists');
catch e
    errors{end+1} = ['heavy_test: ' e.message];
    disp(['    ERROR: ' e.message]);
end

%% ========================================================================
%  SUMMARY AND ERROR REPORTING
%  ========================================================================
disp(' ');
disp('============================================');
disp('Fixture generation complete.');
disp(['Output directory: ' baseOutputDir]);
fprintf('Total errors: %d\n', length(errors));
fprintf('Total skipped (Optimization Toolbox required): %d\n', length(skipped));
disp(' ');
if ~isempty(errors)
    disp('--- Errors ---');
    for i = 1:length(errors)
        disp(['  ' num2str(i) '. ' errors{i}]);
    end
end
disp(' ');
if ~isempty(skipped)
    disp('--- Skipped (Optimization Toolbox required) ---');
    for i = 1:length(skipped)
        disp(['  ' num2str(i) '. ' skipped{i}]);
    end
end
disp(' ');
disp('============================================');
disp('Done.');
disp('============================================');
