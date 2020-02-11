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
R(5,1) = 11;
R(:,2) = 2;


% Find Best Policy under Known True Transitions
[Policy, Q]= MDP_ValueIteration(States, Actions, R, T_True);

%% Experience Replay for Bayesian RL

% inital prior belief
for s = 1:nStates
    for a = 1:nActions
        % Each with uniform distribution
        Priors{a,s} = [1,1,1,1,1];
    end
end

%% Experience Replay for Bayesian RL

% inital prior belief
for s = 1:nStates
    for a = 1:nActions
        % Each with uniform distribution
        Priors{a,s} = [1,1,1,1,1];
    end
end

n_samples = 100;


maxEpisode = 100;
maxIter = 50;

memory = 10;
D = cell(memory,1);
e = 0;
allscores = zeros(maxEpisode,1);
allregrets = zeros(maxEpisode,1);
allpost = cell(maxEpisode,1);
for ep = 1:maxEpisode
    ep
    S = 1;
    t = 1;
    score = 0;
    regret = 0;
    for iter = 1:maxIter
%     while S~=5
        % Take samples for each action
        samples = zeros(nStates,nStates,nActions,n_samples);
        for s = 1:nStates % 
            for a = 1:nActions
                prior = Priors{a,s};
                samples(s,:,a,:) = drchrnd(prior, n_samples)';
            end
        end
    %     samples(:,:,:,1);
        if rand(1) < e
            best_action = randi(Actions);
        else
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
        end
        action = best_action;
        Q_act = Q_all(action,S);
        T_opt = zeros(nStates,nStates,nActions);
        for a = 1:nActions
            T_pri = cell2mat(Priors(a,:)');
            T_opt(:,:,a) = T_pri./sum(T_pri,2);
        end
        [~, Q]= MDP_ValueIteration(States, Actions, R, T_opt);
        
        Q_opt = Q(action,S);
        reg = (Q_act-Q_opt);
        regret = regret + reg;
        
        % Interact with the World; Transition to the next state
        pdf = T_True(S, :, action);
        cdf = cumsum(pdf);
        r = rand(1); % sample an uniform prob
        S_prime = find(r<cdf,1,'first');
        if S_prime == 5
            score = score + 1;
        end 
        % Store the transition minibatch into D memory
        minibatch = [S,action, R(S,action), S_prime];
        D(1) = [];
        D{memory} = minibatch;
        
        % sample from D
        r = randi([memory-t+1, memory]);
        trans = D{r};
        S = trans(1);
        action = trans(2);
        S_prime = trans(4);
        
        Priors{action,S}(S_prime) = Priors{action,S}(S_prime) + 1;
        S = S_prime;
%         S
        t = t+1;
        t = min(t, memory);
        
        
    end
    score
    regret
    allscores(ep) = score;
    allregrets(ep) = regret;
    Post = zeros(nStates,nStates,nActions);
    for a = 1:nActions
        p = cell2mat(Priors(a,:)');
        Post(:,:,a) = p./sum(p,2);
    end
    allpost{ep} = Post;
end





%% Dirichlet Distribution Visualization
alphas = [1,2,4,8,16];
x = 0:0.01:1;
prob = alphas/sum(alphas);

figure()
dirpdf = drchrnd(alphas,200);
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