clc;clear;close all
%{
% States and Actions for Single Agent Fire Extinguish Scenario
States = [1,2,3];  % fire level
nStates = numel(States);
Actions = [1,2];  % 1 move 2 extinguish
nActions = numel(Actions);

% True Transition Probability
T_True = zeros(nStates,nStates,nActions);
% Move
T_True(:,:,1) = [  1,   0,   0;
                 0.9, 0.1,   0;
                 0.5, 0.4, 0.1];
% Stay
T_True(:,:,2) = [   1,   0,   0;
                  0.8, 0.2,   0;
                    0, 0.8, 0.2];
              
% Reward
R = zeros(nStates,nActions);
R(1,1) = 10;
R(1,2) = 10;


% Find Best Policy under Known True Transitions
% [Policy, Q]= MDP_ValueIteration(States, Actions, R, T_True);
%}
%%
% Chain Problem
States = [1,2,3,4,5];  % fire level
nStates = numel(States);
Actions = [1,2];  % 2 actions with different prob move to right
nActions = numel(Actions);

% True Transition Probability
T_True = zeros(nStates,nStates,nActions);
% Move
T_True(:,:,1) = [0.2, 0.8,   0,   0,   0;
                 0.2,   0, 0.8,   0,   0;
                 0.2,   0,   0, 0.8,   0;
                 0.2,   0,   0,   0, 0.8;
                 0.2,   0,   0,   0, 0.8];
% Stay
T_True(:,:,2) = [0.8, 0.2,   0,   0,   0;
                 0.8,   0, 0.2,   0,   0;
                 0.8,   0,   0, 0.2,   0;
                 0.8,   0,   0,   0, 0.2;
                 0.8,   0,   0,   0, 0.2];
              
% Reward
R = zeros(nStates,nActions);
R(5,1) = 10;
R(:,2) = 2;


% Find Best Policy under Known True Transitions
[Policy, Q]= MDP_ValueIteration(States, Actions, R, T_True);

%% Thompson Sampling for Bayesian RL
% inital prior belief
for s = 1:nStates
    for a = 1:nActions
        % Each with normal distribution
        Priors{a,s} = [1,1,1,1,1];
    end
end

S = 1;  % inital state

n_samples = 200;

% while S>0 % online partial planning 
for ii = 1:100
    % Take samples for each action
    samples = zeros(nStates,nStates,nActions,n_samples);
    for s = 1:nStates % 
        for a = 1:nActions
            prior = Priors{a,s};
            samples(s,:,a,:) = drchrnd(prior, n_samples)';
        end
    end
%     samples(:,:,:,1);
    % Get average Q table
    Q_all = zeros(nActions,nStates);
    for n = 1:n_samples
        T = samples(:,:,:,n);
        [Policy, Q]= MDP_ValueIteration(States, Actions, R, T);
        Q_all = Q_all + Q;
    end
    Q_all = Q_all/n_samples;
    % Find the best action for the current state
    [~, best_action] = max(Q_all(:,S));
    
    % Transition to the next state
    pdf = T_True(S, :, best_action);
    cdf = cumsum(pdf);
    r = rand(1); % sample an uniform prob
    S_prime = find(r<cdf,1,'first');
    Priors{best_action,S}(S_prime) = Priors{best_action,S}(S_prime) + 1;
    best_action
%     if S == 5 && S_prime==5
%         break
%     end
    S = S_prime;
end

%% Dirichlet Distribution Visualization
alphas = [1,1,1];
x = 0:0.01:1;


gamma = 0.95;
dirpdf = drchrnd([1,1,1],200);
scatter3(dirpdf(:,1),dirpdf(:,2),dirpdf(:,3))
xlabel('x1')
ylabel('x2')
zlabel('x3')

function r = drchrnd(a,n)
% take a sample from a dirichlet distribution
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);
end