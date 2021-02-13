function [RMSEtrain_end,RMSEtest,t,C,Sigma,W]=OTSK(XTrain,yTrain,XTest,yTest,alpha,beta,lambda1,lambda2,P,nMFs,nIt,C,Sigma)

% This function implements the MBGD-RDA algorithm in the following paper:
%
% Dongrui Wu, Ye Yuan, Jian Huang and Yihua Tan, "Optimize TSK Fuzzy Systems for Regression Problems: 
% Mini-Batch Gradient Descent with Regularization, DropRule and AdaBound (MBGD-RDA)," IEEE Trans. 
% on Fuzzy Systems, 2020, accepted.
%
% It specifies the number of Gaussian MFs in each input domain by nMFs.
% Assume x1 has two MFs X1_1 and X1_2; then, all rules involving the first FS of x1 use the same X1_1,
% and all rules involving the second FS of x1 use the same X1_2
%
% By Dongrui Wu, drwu@hust.edu.cn
%
% %% Inputs:
% XTrain: N*M matrix of the training inputs. N is the number of samples, and M the feature dimensionality.
% yTrain: N*1 vector of the labels for XTrain
% XTest: NTest*M matrix of the test inputs
% yTest: NTest*1 vector of the labels for XTest
% alpha: scalar, learning rate
% rr: scalar, L2 regularization coefficient 
% P: scalar in [0.5, 1), dropRule rate
% nMFs: scalar in [2, 5], number of MFs in each input domain
% nIt: scalar, maximum number of iterations
% Nbs: batch size. typically 32 or 64
%
% %% Outputs:
% RMSEtrain: 1*nIt vector of the training RMSE at different iterations
% RMSEtest: 1*nIt vector of the test RMSE at different iterations
% C: M*nMFs matrix of the centers of the Gaussian MFs
% Sigma: M*nMFs matrix of the standard deviations of the Gaussian MFs
% W: nRules*(M+1) matrix of the consequent parameters for the rules. nRules=nMFs^M.
tic
kkk = 1;
[N,M]=size(XTrain); NTest=size(XTest,1);
% if Nbs>N; Nbs=N; end
nMFsVec=nMFs*ones(M,1);
% nRules=nMFs^M; % number of rules
nRules=30; % number of rules
% C=zeros(M,nMFs); Sigma=C;
W=zeros(nRules,M+1);%W=zeros(nRules,M+1);
% for m=1:M % Initialization
%     C(m,:)=linspace(min(XTrain(:,m)),max(XTrain(:,m)),nMFs);
%     Sigma(m,:)=std(XTrain(:,m));
% end
%% Iterative update
mu=zeros(M,nMFs);
Wz = zeros(nRules,M);Wn = zeros(nRules,M);
for it=1:1
    if nIt <= 1
        Nbs = N;
        f=ones(Nbs,nRules); % firing level of rules
        idsTrain=datasample(1:N,Nbs,'replace',false);
        idsGoodTrain=true(Nbs,1);
    else
        Nbs = nIt*N;
        f=ones(Nbs,nRules); % firing level of rules
        idsTrain = [];idsGoodTrain=true(Nbs,1);
        for i = 1:nIt
            idsTrain=[idsTrain,datasample(1:N,N,'replace',false)];
        end
    end
    yPred=nan(Nbs,1);
    for n=1:Nbs
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(XTrain(idsTrain(n),m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        idsKeep=rand(1,nRules)<=P;
        f(n,~idsKeep)=0;
        for r=1:nRules
            if idsKeep(r)
                idsMFs=idx2vec(r,nMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            end
        end
        if ~sum(f(n,:)) % special case: all f(n,:)=0; no dropRule
            idsKeep=true(1,nRules);
            f(n,:)=1;
            for r=1:nRules
                idsMFs=idx2vec(r,nMFsVec);
                for m=1:M
                    f(n,r)=f(n,r)*mu(m,idsMFs(m));
                end
            end
        end
        fBar=f(n,:)/sum(f(n,:));
%         yR=[XTrain(idsTrain(n),:)]*W';
        yR=[1 XTrain(idsTrain(n),:)]*W';
        yPred(n)=fBar*yR'; % prediction
        if isnan(yPred(n))
            %save2base();          return;
            idsGoodTrain(n)=false;
            continue;
        end
        % the core 
        % Compute delta
        for r=1:nRules
            if idsKeep(r)
                temp=(yPred(n)-yTrain(idsTrain(n)))*(yR(r)*sum(f(n,:))-f(n,:)*yR')/sum(f(n,:))^2*f(n,r);
                if ~isnan(temp) && abs(temp)<inf
                    vec=idx2vec(r,nMFsVec);
                    % delta of c, sigma, and b
                    %for parameter C, bo,and Sigma
                    tol = 1/sqrt(n);
                    for m = 1:M
                        C(m,vec(m))=C(m,vec(m))-tol*temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))/Sigma(m,vec(m))^2;
                        Sigma(m,vec(m))=Sigma(m,vec(m))-tol*temp*(XTrain(idsTrain(n),m)-C(m,vec(m)))^2/Sigma(m,vec(m))^3;
                    end
                    W(r,1)=W(r,1)-tol*(yPred(n)-yTrain(idsTrain(n)))*fBar(r);
                    I = find(XTrain(idsTrain(n),:)~=0);
                    %for parameter W
                    for j = 1:length(I)
                        a = I(j);
                        if abs(Wz(r,a)) <= lambda1
                            W(r,a+1) = 0;
                        else
                            W(r,a+1) = -1/((beta + sqrt(Wn(r,a)))/alpha + lambda2)*(Wz(r,a) - sign(Wz(r,a))*lambda1);
                        end   
                    end
                    % update g, n, z
                    for j = 1:length(I)
                        a = I(j);
                        g = (yPred(n)-yTrain(idsTrain(n)))*fBar(r)*XTrain(idsTrain(n),a);
                        sigma = (sqrt(Wn(r,a) + g*g) -sqrt(Wn(r,a)))/alpha;
                        Wz(r,a) = Wz(r,a)+g-sigma*W(r,a+1);
                        Wn(r,a) = Wn(r,a)+g*g;
                    end
                end
            end
        end
%     end
    % Test RMSE
    if (~mod(n,500))
    f=ones(NTest,nRules); % firing level of rules
    for i=1:NTest
        for m=1:M % membership grades of MFs
            mu(m,:)=exp(-(XTest(i,m)-C(m,:)).^2./(2*Sigma(m,:).^2));
        end
        for r=1:nRules % firing levels of rules
            idsMFs=idx2vec(r,nMFsVec);
            for m=1:M
                f(i,r)=f(i,r)*mu(m,idsMFs(m));
            end
        end
    end
    yR=[ones(NTest,1) XTest]*W';
    yPredTest=sum(f.*yR,2)./sum(f,2); % prediction
    RMSEtest(kkk)=sqrt((yTest-yPredTest)'*(yTest-yPredTest)/NTest);
    if isnan(RMSEtest(kkk)) && kkk>1
        RMSEtest(kkk)=RMSEtest(kkk-1);
    end
    % Training RMSE
%     RMSEtrain(n)=sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/n);
    t(kkk) = toc;
    kkk = kkk+1;
    end
    end
    temp1 = min(RMSEtest);
    RMSEtest_end = temp1(1);
%     % Training RMSE
    RMSEtrain_end=sqrt(sum((yTrain(idsTrain(idsGoodTrain))-yPred(idsGoodTrain)).^2)/sum(idsGoodTrain));
end
end

function vec=idx2vec(idx,nMFs)
% Convert from a scalar index of the rule to a vector index of MFs
vec=zeros(1,length(nMFs));
prods=[1; cumprod(nMFs(end:-1:1))];
if idx>prods(end)
    error('Error: idx is larger than the number of rules.');
end
prev=0;
for i=1:length(nMFs)
    vec(i)=floor((idx-1-prev)/prods(end-i))+1;
    prev=prev+(vec(i)-1)*prods(end-i);
end
end