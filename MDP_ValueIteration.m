function [Policy, Q] = MDP_ValueIteration(S,A,R,T)
nStates = numel(S);
nActions = numel(A);
gamma = 0.95;
epsilon = 1;
V{1} = 0*ones(1,nStates);  % initial value for each state
t = 1;
% tic
while t
    t = t+1;
    % for all states
    for s = 1:nStates
        % Reward at this state
        for a = 1:nActions
        % for all actions
            % get transition probability to next states
            TransProb = T(s,:,a);
            s_prime.idx = find(TransProb>0);
            s_prime.transProb = TransProb(s_prime.idx);
            % sum for all s' T(s,a,s')Vt-1(s')
            nextValues = V{t-1}([s_prime.idx])';
            transProbToNext = [s_prime.transProb];
            propagateValue = transProbToNext*nextValues;
            % Compute Q
            reward = R(s,a);
            Q(t,a,s) = reward + gamma * propagateValue;
        end
        V{t}(s) = max(Q(t,:,s));
    end
    
    error = max(V{t}-V{t-1});
    if max(V{t}-V{t-1})< epsilon
        error = max(V{t}-V{t-1});
        break
    end
end
% toc

% Find Best Policy
Vend = V{end};
Policy = {};
for s = 1:nStates
    ExpectedUtility = [];
    for a = 1:nActions
        TransProb = T(s,:,a);
        s_prime.idx = find(TransProb>0);
        s_prime.transProb = TransProb(s_prime.idx);
        nextValues = Vend([s_prime.idx])';
        transProbToNext = [s_prime.transProb];
        expectedUtility = transProbToNext*nextValues;
        ExpectedUtility = [ExpectedUtility expectedUtility];
    end
    [~, i] = max(ExpectedUtility);
    Policy{s} = i;
%     disp(['State: ' num2str(s) ', Action: ' num2str(i)])
end
QQ(:,:) = Q(end,:,:);
Q = QQ;
end

