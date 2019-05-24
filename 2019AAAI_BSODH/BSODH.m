clear;
opts.dirs.data = '/home/lmb/source/hash_baseline/mihash-master/data';
opts.unsupervised = 0;
opts.metric = 'mAP';
opts.nbits = 32;
normalizeX = 1;


%DS = Datasets.places(opts, normalizeX);
DS = Datasets.cifar(opts, normalizeX);
%DS = Datasets.mnist(opts, normalizeX);

trainCNN = DS.Xtrain; % n x  d
testCNN = DS.Xtest;
trainLabels = DS.Ytrain; %  n x d
testLabels = DS.Ytest;

% mapped into a sphere space
test = testCNN ./ sqrt(sum(testCNN .* testCNN, 2));  
%test = testCNN;
testLabel = testLabels;  % n x 1
train = trainCNN ./ sqrt(sum(trainCNN .* trainCNN, 2));   
%train = trainCNN;
trainLabel = trainLabels; % n x 1
clear testCNN trainCNN testLabels trainLabels

[Ntrain, Dtrain] = size(train);
[Ntest, Dtest] = size(test);

test = test';
train = train';


W_t = randn(Dtest, opts.nbits);
W_t = W_t ./ repmat(diag(sqrt(W_t' * W_t))', Dtest, 1);


%%%%%%%%%%%% Four parameters depicted in the paper %%%%%%%%%%%%%%%%
lambda = 0.3;
sigma = 0.5;
etad = 0.2;
etas = 1.2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_t = 2000;              % training size at each stage
training_size = 20000;   % total training instances


Xs_t = [];
Bs_t = [];
ls_t = [];

Be_t = [];
Xe_t = [];
le_t = [];

S_t = [];


tic
for t = n_t:n_t:training_size
    if t == n_t       % first stage
        Xe_t = train(:, 1 : n_t);
        tmp = W_t' * Xe_t;
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;
        Be_t = tmp;

        le_t = trainLabel(1 : n_t);
        continue;
    end
    if t + n_t > Ntrain
        break
    end
    Xe_t = [Xe_t, Xs_t];
    Be_t = [Be_t, Bs_t];
    le_t = [le_t; ls_t];


    Xs_t = train(:, t - n_t + 1 : t);

    tmp = W_t' * Xs_t;
    tmp(tmp >= 0) = 1;
    tmp(tmp < 0) = -1;

    Bs_t = tmp;  
    ls_t = trainLabel(t - n_t + 1 : t);
    S_t = single(ls_t == le_t');    % it can be easily extended to the multi-label case
    for i = 1:n_t
        if sum(S_t(i,:)) ~= 0
            ind = find(S_t(i,:) ~=0);
            Bs_t(:, i) = Be_t(:, ind(1));
        end
    end

    S_t(S_t == 0) = -etad;
    S_t(S_t == 1) = etas;

    S_t = S_t * opts.nbits;


    tag = 1;

    % update Bs
    P = opts.nbits * Be_t * S_t' + W_t' * Xs_t;
    for r = 1:opts.nbits
        be = Be_t(r, :);
        Be_hat = [Be_t(1:(r-1),:); Be_t((r+1):end, :)];

        bs = Bs_t(r, :);
        Bs_hat = [Bs_t(1:(r-1),:); Bs_t((r+1):end, :)];

        p = P(r, :);
        P_hat = [P(1:(r-1), :); P((r+1):end, :)];

        tmp = p - be * Be_hat' * Bs_hat;
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;

        Bs_t(r, :) = tmp;
    end

    % update Be
    Be_t = Bs_t * S_t;
    Be_t(Be_t >= 0) = 1;
    Be_t(Be_t < 0) = -1;

    % update W
    I = eye(Dtrain);
    W_t = sigma * inv(sigma * Xs_t * Xs_t' + lambda * I) * Xs_t * Bs_t';
end
toc

Htrain = single(W_t' * train > 0);

Htest = single(W_t' * test > 0);

Aff = affinity([], [], trainLabel, testLabel, opts);

res = evaluate(Htrain', Htest', opts, Aff);

clear;

