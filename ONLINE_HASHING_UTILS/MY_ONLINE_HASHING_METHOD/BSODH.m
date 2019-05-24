clear;
opts.dirs.data = '../data';
opts.unsupervised = 0;
opts.nbits = 32;
normalizeX = 1;


%DS = Datasets.places(opts, normalizeX);
DS = Datasets.cifar(opts, normalizeX);
%DS = Datasets.mnist(opts, normalizeX);

trainCNN = DS.Xtrain;  % n x  d
testCNN = DS.Xtest;
trainLabels = DS.Ytrain;  %  n x d
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
lambda = 0.3;   %  0.6 for CIFAR-10, 0.3 for MNIST, 0.9 for Places205
sigma = 0.5;    %  0.5 for CIFAR-10, 0.5 for MNIST, 0.8 for Places205
etad = 0.2;     %  0.2 for CIFAR-10 and MNIST, 0 for Places205
etas = 1.2;     %  1.2 for CIFAR-10 and MNIST, 1 for Places205
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_t = 2000;     % training size at each stage       % 2K for CIFAR-10 and MNIST, 10K for Places205
training_size = 20000;   % total training instances % 20K for CIFAR-10 and MNIST, 100K for Places205


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

opts.metric = 'mAP';
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'mAP_';
opts.mAP = 1000;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_n2';
opts.prec_n = 2;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 1;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 5;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 10;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 20;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 30;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 40;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 50;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 60;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 70;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 80;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 90;
res = evaluate(Htrain', Htest', opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 100;
res = evaluate(Htrain', Htest', opts, Aff);

clear;
