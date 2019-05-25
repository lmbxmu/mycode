% Supervised Online Hashing via Hadamard Codebook Learning, ACM MM 2018.

clear;
opts.dirs.data = '../data'
opts.unsupervised = 0;
opts.nbits = 32;
hbits = 32; %  the length of Hadamard codebook. It has to be min{2^k, unique(train_label)}.
normalizeX = 1; 

%DS = Datasets.places(opts, normalizeX);
DS = Datasets.cifar(opts, normalizeX);
%DS = Datasets.mnist(opts, normalizeX);

trainCNN = DS.Xtrain;    % n x d
testCNN = DS.Xtest;      % n x d
trainlabel = DS.Ytrain;
testlabel = DS.Ytest;


[Ntrain, Dtrain] = size(trainCNN);
[Ntest, Dtest] = size(testCNN);


h = hadamard(hbits);      % Hadamard Matrix
h = h(randperm(hbits), :);

train_label = h(trainlabel, :);  % assign Hadamard codebook based on the label.

% mapping the length of Hadamard codebook to the code length.
lshW = randn(hbits, opts.nbits);
lshW = lshW ./ repmat(diag(sqrt(lshW'*lshW))', hbits, 1);
% hash weight
W = randn(Dtrain, opts.nbits);
W = W ./ repmat(diag(sqrt(W'* W))', Dtrain, 1);


%%%%%%%%%%%%%%%Parameter used in the paper%%%%%%%%%%%%%%%%%%%%
n_t = 1;    % training size at each stage   % 1 for CIFAR-10, MNIST and Places205
eta = 0.2;  % learning rate                 % 0.2 for CIFAR-10 and MNIST, 0.1 for Places205
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

training_size = 20000;   % total training instances % 20K for CIFAR-10 and MNIST, 100K for Places205


tic
for t = 1:n_t:training_size
    B = train_label(t:(t+n_t-1), :);
    X = trainCNN(t:(t+n_t-1), :);
    if hbits ~= opts.nbits
        B = single(B * lshW > 0);
        B(B<=0) = -1;
    end

    F = tanh(X*W);
    der = eta * X' * [(F - B) .* (1 - F.*F)] / n_t;
    W = W - der;
end
toc


Htrain = single(trainCNN * W > 0);

Htest = single(testCNN * W > 0);

Aff = affinity([], [], trainlabel, testlabel, opts);

opts.metric = 'mAP';
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'mAP_';
opts.mAP = 1000;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_n2';
opts.prec_n = 2;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 1;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 5;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 10;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 20;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 30;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 40;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 50;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 60;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 70;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 80;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 90;
res = evaluate(Htrain, Htest, opts, Aff);

opts.metric = 'prec_k1';
opts.prec_k = 100;
res = evaluate(Htrain, Htest, opts, Aff);

clear;
