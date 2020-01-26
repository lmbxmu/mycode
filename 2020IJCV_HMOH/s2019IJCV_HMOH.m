clear;
opts.dirs.data = '../data'
opts.unsupervised = 0;
opts.metric = 'mAP';
normalizeX = 0; % CIFAR-10, Places205, MNIST: 0; NUS-WIDE: 0




%DS = Datasets.mnist(opts, normalizeX);     
DS = Datasets.cifar(opts, normalizeX);     
%DS = Datasets.places(opts, normalizeX);   
%DS = Datasets.nuswide(opts, normalizeX);

trainCNN = DS.Xtrain;
testCNN = DS.Xtest;
trainlabel = DS.Ytrain;
testlabel = DS.Ytest;

num = 1; % For all datasets
lr = 0.1;  % MNIST: 0.1, Places: 0.5, CIFAR-10: 0.01, NUS-WIDE:0.1
eta = 280;   % MNIST: 17.75, Places205: 692, nus-wide: 280
training_size = 20000;  %  CIFAR-10 and MNIST: 20000, Places205: 100000, NUS-WIDE: 40000
kernel_size = 500; % MNIST: 300, Places: 800, NUS-WIDE: 500

nhalf = floor(size(trainCNN, 1)/2);
ind = randperm(nhalf, kernel_size);
Xanchor = trainCNN(ind, :);
% 
ind = randperm(nhalf, 2000);
Xval = trainCNN(nhalf+ind, :);
Kval = sqdist(Xval', Xanchor');
eta = mean(mean(Kval, 2))/10; % mnist:6, places:10, nus-wide: 10

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do not apply the following two lines on CIFAR-10 and NUS-WIDE (i.e., no kernelization)
%trainCNN = exp(-sqdist(trainCNN', Xanchor')/(2*eta));
%testCNN = exp(-sqdist(testCNN', Xanchor')/(2*eta));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[Ntrain, Dtrain] = size(trainCNN);
[Ntest, Dtest] = size(testCNN);



bit = [8,16,32,48,64,128];
hbit = [16,16,32,64,64,128];  % For MNIST and CIFAR-10
%hbit = [256, 256, 256, 256, 256, 256]; % For Places205
bit = [32];
hbit = [32];

    
for i = 1:length(bit)
    opts.nbits = bit(i);
    hbits = hbit(i);
    h = hadamard(hbits);
    
    if size(trainlabel, 2) == 1  % single-label
        train_label = h(trainlabel, :);
    else   % multi-label
        label_tmp = [];
        for j = 1:training_size
            tmp = sum(h(trainlabel(j,:) == 1, :), 1);
            tmp(tmp > 0) = 1;
            cnt_pos = sum(tmp == 1);
            tmp(tmp < 0) = -1;
            cnt_neg = sum(tmp == -1);
            
            ind_zero = find(tmp == 0);
            ind_zero = ind_zero(randperm(length(ind_zero)));
            
            dif = abs(cnt_pos - cnt_neg);
            nzero = length(ind_zero);
            
            if nzero ~= 0
                minj = max(dif, nzero);
            
                if cnt_pos > cnt_neg
                    tmp(ind_zero(1:minj)) = -1;
                    ind_zero = ind_zero(minj + 1:end);
                end
                if cnt_pos < cnt_neg
                    tmp(ind_zero(1:minj)) = 1;
                    ind_zero = ind_zero(minj + 1:end);
                end

                if length(ind_zero) > 0
                    mid = round(length(ind_zero)/2);
                    tmp(1:mid) = 1;
                    tmp(mid+1:end) = -1;
                end 
%                sum(tmp) 
            end  
            label_tmp = [label_tmp; tmp];             
        end    
        train_label = label_tmp;
    end
        

    
    


    lshW = randn(hbits, opts.nbits);
    lshW = lshW ./ repmat(diag(sqrt(lshW'*lshW))', hbits, 1);


    W = randn(Dtrain, opts.nbits);
    W = W ./ repmat(diag(sqrt(W'* W))', Dtrain, 1);


    W = zeros(Dtrain, opts.nbits);
    tmp_W = 0;


    cnt = zeros(1, opts.nbits);


    tic
    for t = 1:num:training_size
        train_class_hash = train_label(t:(t+num-1), :);

        X = trainCNN(t:(t+num-1), :);


        f = X*W;
        F = sign(f);
        F(F == 0) = -1;

        if opts.nbits ~= hbits
            B1 = single(train_class_hash * lshW > 0);
            B1(B1<=0) = -1;     
        else
            B1 = train_class_hash;
        end

    
    % Perceptual Algorithm
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        xx = (F~=B1);
        T = single(xx); % false classification
        cnt(xx) = cnt(xx) + 1;
        T = T .* B1;
        T = reshape(T', [1, opts.nbits, num]);
        T_mat = repmat(T, [Dtrain, 1, 1]);

        X_hat = reshape(X', [Dtrain, 1, num]);
        X_mat = repmat(X_hat, [1, opts.nbits, 1]);
        tmp = T_mat .* X_mat;
        tmp = sum(tmp, 3)/num;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        W = W + lr*tmp;
        tmp_W = tmp_W + W;
        
    end
    
    tmp_W = tmp_W ./ repmat(cnt, Dtrain, 1);  % averaged
    
    train_time = toc;
    logInfo(['train time = ' num2str(train_time)]);

    
    if ~exist('Aff')
        Aff = affinity([], [], trainlabel, testlabel, opts);
    end
    
    Htrain = single(trainCNN * tmp_W > 0);
    Htest = single(testCNN * tmp_W > 0);
    TRAIN = [Htrain];
    TEST = [Htest];
    opts.metric = 'mAP';
    opts.mAP = 1000;
    res = evaluate(TRAIN, TEST, opts, Aff);
    
    opts.metric = 'prec_n2';
    opts.prec_n = 2;
%    res = evaluate(TRAIN, TEST, opts, Aff);  
    
end
