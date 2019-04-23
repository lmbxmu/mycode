clear;
opts.dirs.data = '/home/lmb/source/hash_baseline/mihash-master/data';
opts.unsupervised = 0;
%opts.metric = 'mAP_1000';
%opts.metric = 'mAP';
opts.metric = 'prec_n2';
opts.prec_n = 2;
%opts.mAP = 1000;
opts.nbits = 128;
normalizeX = 0;


DS = Datasets.cifar(opts, normalizeX);
trainCNN = DS.Xtrain;
testCNN = DS.Xtest;
trainLabels = DS.Ytrain;
testLabels = DS.Ytest;

% mapped into a sphere space
test = testCNN ./ sqrt(sum(testCNN .* testCNN, 2));  % n x  d
%test = testCNN;
testLabel = testLabels;  % n x 1
train = trainCNN ./ sqrt(sum(trainCNN .* trainCNN, 2));   %  n x d
%train = trainCNN;
trainLabel = trainLabels; % n x 1
clear testCNN trainCNN testLabels trainLabels

[Ntrain, Dtrain] = size(train);
[Ntest, Dtest] = size(test);

n_t = 2000;
test = test';
train = train';


training_data = [20000];
tic
for xx = 1:1
    training_size = training_data(xx);
%    training_size = 100000;


    W_t = randn(Dtest, opts.nbits);
    W_t = W_t ./ repmat(diag(sqrt(W_t' * W_t))', Dtest, 1);


    sigma = 0.5;
    lambda = 0.3;
    epoch = 1;
    I = eye(opts.nbits);
    %training_size = 20000;


    Xs_t = [];
    Bs_t = [];
    ls_t = [];

    Be_t = [];
    Xe_t = [];
    le_t = [];


    S_t = [];


    time = 0;
    for t = n_t:n_t:training_size
        time = time + 1;
        if t == n_t
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

    %    lshW = randn(Dtrain, opts.nbits);
    %    lshW = lshW ./ repmat(diag(sqrt(lshW'*lshW))', Dtrain, 1);
    %    tmp = lshW' * Xs_t;
        tmp = W_t' * Xs_t;
        tmp(tmp >= 0) = 1;
        tmp(tmp < 0) = -1;

        Bs_t = tmp;  
        ls_t = trainLabel(t - n_t + 1 : t);
        S_t = single(ls_t == le_t');
        for i = 1:n_t
            if sum(S_t(i,:)) ~= 0
                ind = find(S_t(i,:) ~=0);
                Bs_t(:, i) = Be_t(:, ind(1));
            end
        end


        S_t(S_t == 0) = 0.2;
        S_t(S_t == 1) = 1.2;
    %      total = length(S_i(:));
    %      tp = (S_i == 1);
    %      pos = sum(tp(:));
    %      tn = (S_i ~= 1);
    %      neg = sum(tn(:));
    %      S_i(S_i == 0) = -pos / total;
    %      S_i(S_i == 1) = neg / total;
    %      pos / total
    %      neg / total
         S_t = S_t * opts.nbits;


        tag = 1;
        for j = 1:epoch
        %    i, j

            % update Bs
            tmp_Bs = Bs_t;
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

                if sum(tmp ~= Bs_t(r, :)) > 0
                    Bs_t(r, :) = tmp;
                    tag = 1;
                end

            end
            err = sum(sum(abs(tmp_Bs - Bs_t)));
    %        comb = [time, j, err]

            % update Be
            Be_t = Bs_t * S_t;
            Be_t(Be_t >= 0) = 1;
            Be_t(Be_t < 0) = -1;

            % update W
            la = 0.0;
            I = eye(Dtrain);
            if tag == 1
                W_t = sigma * inv(sigma * Xs_t * Xs_t' + lambda * I) * Xs_t * Bs_t';
    %             F = tanh(W_t' * Xs_t);
    %             derivative = lr1 * Xs_t * ((F - Bs_t) .* (1 - F.*F))'+ la*W_t*(W_t'*W_t - eye(opts.nbits));
    %             W_t = W_t - derivative;
                tag = 0;
            end



        end

    end


%     Htrain = single(W_t' * train > 0);
%     B = [Be_t, Bs_t];
% 
%     Htest = single(W_t' * test > 0);
% 
%     Aff = affinity([], [], trainLabel, testLabel, opts);
% 
%     opts.metric = 'mAP_1000';
%     opts.mAP = 1000;
%    res = evaluate(Htrain', Htest', opts, Aff);
end
toc

Htrain = single(W_t' * DS.Xretrieval' > 0);

Htest = single(W_t' * test > 0);

Aff = affinity([], [], double(DS.Yretrieval), testLabel, opts);


opts.metric = 'prec_k1';
opts.prec_k = 1;
res = evaluate2(Htrain', Htest', opts, Aff);


