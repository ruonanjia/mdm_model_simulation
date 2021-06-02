clearvars
close all

%% trial set
SAFE_REWARD = 5;
% AMBIGUITIES = [25, 50, 75]./100;
AMBIGUITIES = [24, 50, 74, 1]./100;
% PROBABILITIES = [24, 50, 74]./100;
PROBABILITIES = [25, 50, 75]./100;
RISKY_REWARDS = [5, 8, 12, 25]; % Don't include 5
[P, V_P] = ndgrid(PROBABILITIES,RISKY_REWARDS); % RISKY problem set
[A, V_A] = ndgrid(AMBIGUITIES,RISKY_REWARDS);  % AMBIGUITY problem set
P = reshape(P, [], 1);
V_P = reshape(V_P, [], 1);
A = reshape(A, [], 1);
V_A = reshape(V_A, [], 1);

% each trial repeated four times
values = repmat([V_P; V_A], 4, 1);
probs = repmat([P; 0.5*ones(size(A))], 4, 1);
ambigs = repmat([zeros(size(P)); A], 4, 1);

values_fixed = 5*ones(size(values));
probs_fixed = ones(size(values));

n_trials = length(values);

gamma_fixed = -2.6;

batch = '06022021';
path_save = fullfile('E:/Ruonan/Projects in the lab/MDM Project/Medical Decision Making Imaging/MDM_imaging/Behavioral Analysis/simulation_results', batch);

if ~exist(path_save)
    mkdir(path_save);
end

%% generate behaviorl from repetitions
model_true = 'ambigNrisk';
base = 0;

MIN_PARAM_alpha = 0.2; MAX_PARAM_alpha = 1.5;
MIN_PARAM_beta = -1.5; MAX_PARAM_beta = 1.5;
get_alpha = @(n) MIN_PARAM_alpha + (rand(1, n) * (MAX_PARAM_alpha-MIN_PARAM_alpha)); %uniform
get_beta = @(n) MIN_PARAM_beta + (rand(1, n) * (MAX_PARAM_beta-MIN_PARAM_beta)); %uniform

REPETITIONS = 100;

ids = [1:REPETITIONS];
% generate parameters
alphas = get_alpha(REPETITIONS);
% alphas = repmat(0.5, 1, 100); % fix alphas
betas = get_beta(REPETITIONS);
% betas = 1.8 *(alphas + rand(size(alphas))-2.2) ;
% betas = repmat(0.35, 1, 100); % fix betas
gammas = gamma_fixed*ones(size(alphas)); % Stochastisity of preference

choice_simulated = zeros(REPETITIONS*n_trials, 9);

% generate behavior
for ii = 1:REPETITIONS
    id = ids(ii);
    alpha = alphas(ii);
    beta = betas(ii);
    gamma = gammas(ii);
    
    choice_prob = choice_prob_ambigNrisk(...
        base, values_fixed, values,...
        probs_fixed, probs, ambigs,...
        [gamma, beta, alpha], model_true);

    choice = binornd(1, choice_prob);
    
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 1) = id;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 2) = probs;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 3) = ambigs;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 4) = values;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 5) = alpha;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 6) = beta;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 7) = gamma;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 8) = choice_prob;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 9) = choice;
    
end

choice_simulated = array2table(choice_simulated,...
    'VariableNames', {'id','probs','ambigs', 'values',...
    'alhpa', 'beta','gamma', 'choice_prob', 'choice'});

simulation = struct(...
    'choice', choice_simulated,...
    'model', model_true,...
    'alpha_range', [MIN_PARAM_alpha, MAX_PARAM_alpha],...
    'beta_range', [MIN_PARAM_beta, MAX_PARAM_beta],...
    'distribution', 'uniform');

save(fullfile(path_save, 'simulation_ambigNrisk.mat'), 'simulation');

%% generate behavior from repetitions
model_true = 'ambigOnly';
base = 0;

MIN_PARAM_alpha1 = 0.2; MAX_PARAM_alpha1 = 1.5;
MIN_PARAM_beta = -1.5; MAX_PARAM_beta = 1.5;
get_alpha1 = @(n) MIN_PARAM_alpha1 + (rand(1, n) * (MAX_PARAM_alpha1-MIN_PARAM_alpha1)); %uniform
get_beta = @(n) MIN_PARAM_beta + (rand(1, n) * (MAX_PARAM_beta-MIN_PARAM_beta)); %uniform

REPETITIONS = 100;

ids = [1:REPETITIONS];
% generate parameters
betas = get_beta(REPETITIONS);
gammas = gamma_fixed*ones(size(betas)); % Stochastisity of preference

% how to generate ratings?
% assume a curvature alpha1 from objective value to rating
% another curvature alpha2 from rating to subjective value
alpha1s = get_alpha1(REPETITIONS); % value-rating curvature

choice_simulated = zeros(REPETITIONS*n_trials, 10);

% generate behavior
for ii = 1:REPETITIONS
    id = ids(ii);
    alpha1 = alpha1s(ii);
    beta = betas(ii);
    gamma = gammas(ii);
    
    % generate ratings
    ratings = values .^ alpha1; % generate trial-wise rating of a single subject
    
    % scale ratings to 0-100
    rating_max = max(ratings);
    k = 100/rating_max;
    ratings = ratings .* k;
    ratings_fixed = (values_fixed .^ alpha1) .* k;
    
    choice_prob = choice_prob_ambigOnly(...
        base, ratings_fixed, ratings,...
        probs_fixed, probs, ambigs,...
        [gamma, beta], model_true);

    choice = binornd(1, choice_prob);
    
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 1) = id;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 2) = probs;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 3) = ambigs;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 4) = values;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 5) = ratings;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 6) = alpha1;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 7) = beta;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 8) = gamma;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 9) = choice_prob;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 10) = choice;
    
end

choice_simulated = array2table(choice_simulated,...
    'VariableNames', {'id','probs','ambigs', 'values','ratings',...
    'alhpa1', 'beta','gamma', 'choice_prob', 'choice'});

simulation = struct(...
    'choice', choice_simulated,...
    'model', model_true,...
    'alpha1_range', [MIN_PARAM_alpha1, MAX_PARAM_alpha1],...
    'beta_range', [MIN_PARAM_beta, MAX_PARAM_beta],...
    'distribution', 'uniform');

save(fullfile(path_save,'simulation_ambigOnly.mat'), 'simulation');

%% generate behavior from repetitions
model_true = 'ambigNrisk_rating';
base = 0;

MIN_PARAM_alpha1 = 0.2; MAX_PARAM_alpha1 = 1.5;
MIN_PARAM_alpha2 = 0.2; MAX_PARAM_alpha2 = 1.5;
MIN_PARAM_beta = -1.5; MAX_PARAM_beta = 1.5;
get_alpha1 = @(n) MIN_PARAM_alpha1 + (rand(1, n) * (MAX_PARAM_alpha1-MIN_PARAM_alpha1)); %uniform
get_alpha2 = @(n) MIN_PARAM_alpha2 + (rand(1, n) * (MAX_PARAM_alpha2-MIN_PARAM_alpha2)); %uniform
get_beta = @(n) MIN_PARAM_beta + (rand(1, n) * (MAX_PARAM_beta-MIN_PARAM_beta)); %uniform

REPETITIONS = 100;

ids = [1:REPETITIONS];
% generate parameters
betas = get_beta(REPETITIONS);
gammas = gamma_fixed*ones(size(betas)); % Stochastisity of preference

% how to generate ratings?
% assume a curvature alpha1 from objective value to rating
% another curvature alpha2 from rating to subjective value
alpha1s = get_alpha1(REPETITIONS); % value-rating curvature
alpha2s = get_alpha2(REPETITIONS); % value-rating curvature

choice_simulated = zeros(REPETITIONS*n_trials, 11);

% generate behavior
for ii = 1:REPETITIONS
    id = ids(ii);
    alpha1 = alpha1s(ii);
    alpha2 = alpha2s(ii);
    beta = betas(ii);
    gamma = gammas(ii);
    
    % generate ratings
    ratings = values .^ alpha1; % generate trial-wise rating of a single subject
    
    % scale ratings to 0-100
    rating_max = max(ratings);
    k = 100/rating_max;
    ratings = ratings .* k;
    ratings_fixed = (values_fixed .^ alpha1) .* k;
    
    choice_prob = choice_prob_ambigNrisk(...
        base, ratings_fixed, ratings,...
        probs_fixed, probs, ambigs,...
        [gamma, beta, alpha2], 'ambigNrisk');

    choice = binornd(1, choice_prob);
    
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 1) = id;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 2) = probs;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 3) = ambigs;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 4) = values;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 5) = ratings;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 6) = alpha1;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 7) = alpha2;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 8) = beta;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 9) = gamma;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 10) = choice_prob;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 11) = choice;
    
end

choice_simulated = array2table(choice_simulated,...
    'VariableNames', {'id','probs','ambigs', 'values','ratings',...
    'alhpa1', 'alpha2', 'beta','gamma', 'choice_prob', 'choice'});

simulation = struct(...
    'choice', choice_simulated,...
    'model', model_true,...
    'alpha1_range', [MIN_PARAM_alpha1, MAX_PARAM_alpha1],...
    'alpha2_range', [MIN_PARAM_alpha2, MAX_PARAM_alpha2],...
    'beta_range', [MIN_PARAM_beta, MAX_PARAM_beta],...
    'distribution', 'uniform');

save(fullfile(path_save,'simulation_ambigOnly_rating.mat'), 'simulation');

%% generate behavior from repetitions
model_true = 'ambigSVPar';
base = 0;

MIN_PARAM_beta = -1.5; MAX_PARAM_beta = 1.5;
get_beta = @(n) MIN_PARAM_beta + (rand(1, n) * (MAX_PARAM_beta-MIN_PARAM_beta)); %uniform
get_sv = @(max_sv) sort(max_sv .* rand(1, 4)); % uniformly generate random monotonic numbers


REPETITIONS = 100;
up_bound = 100; % subjective value upper bound

ids = [1:REPETITIONS];
% generate parameters
betas = get_beta(REPETITIONS);
gammas = gamma_fixed*ones(size(betas)); % Stochastisity of preference

% how to generate SVs? from curvature?
% assume a curvature alpha1 from objective value to rating
% another curvature alpha2 from rating to subjective value
% alpha1 * alpha2 = alpha
% alpha1s = get_alpha1(REPETITIONS); % value-rating curvature
% alpha2s = get_alpha2(REPETITIONS); % value-rating curvature

choice_simulated = zeros(REPETITIONS*n_trials, 9);

% generate behavior
for ii = 1:REPETITIONS
    id = ids(ii);
    beta = betas(ii);
    gamma = gammas(ii);
    
    % generate subjective values
    svs = get_sv(up_bound);
    
    choice_prob = choice_prob_ambigNriskValPar(...
        base, values_fixed, values,...
        probs_fixed, probs, ambigs,...
        [gamma, beta, svs], 'ambigSVPar', RISKY_REWARDS);
    
    choice = binornd(1, choice_prob);
    
    % create trial-wise subjective values 
    [~, value_par_idx] = ismember(values, RISKY_REWARDS);
    % get the value parameter
    svs_trialwise = svs(value_par_idx)'; % change dimeinsion into 1 by n

    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 1) = id;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 2) = probs;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 3) = ambigs;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 4) = values;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 5) = svs_trialwise;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 6) = beta;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 7) = gamma;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 8) = choice_prob;
    choice_simulated((n_trials*(ii-1)+1):(n_trials*ii), 9) = choice;
    
end

choice_simulated = array2table(choice_simulated,...
    'VariableNames', {'id','probs','ambigs', 'values','svs',...
    'beta','gamma', 'choice_prob', 'choice'});

simulation = struct(...
    'choice', choice_simulated,...
    'model', model_true,...
    'svs_up_bound', 'up_bound',...
    'beta_range', [MIN_PARAM_beta, MAX_PARAM_beta],...
    'distribution', 'uniform');

save(fullfile(path_save,'simulation_ambigSVPar.mat'), 'simulation');

%% generate behavior from single simulation
% % original model
% model_true = 'ambigNrisk';
% % single subject risk and ambiguity correlation
% alpha = 0.5;
% beta = 1;
% gamma = 2;
% base = 0;
% 
% choice_prob = choice_prob_ambigNrisk(...
%     base, values_fixed, values,...
%     probs_fixed, probs, ambigs,...
%     [gamma, beta, alpha], model_true);
% 
% choice = binornd(1, choice_prob);
