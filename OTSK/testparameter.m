%% Illustrate how MBGD_RDA, MBGD_RDA2, MBGD_RDA_T and MBGD_RDA2_T are used.
%% By Dongrui WU, drwu@hust.edu.cn
clc; clearvars; close all; %rng(0);
LAMBDA = [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001];
nMFs=2; % number of MFs in each input domain, used in MBGD_RDA and MBGD_RDA_T
% the important parameters for FTRL
lambda2 = 1;
nIt=1; % number of iterations
% Nbs1=N0; % batch size
% the important parameters for MBGD
P=0.5; % DropRule rate
PP = [5,10];
for i = 5:5
    if i == 1
        temp=load('NO2.mat');
        data=temp.data;
    elseif i==2
         temp=load('winequality-red.mat'); 
         data = temp.winequality_red;
    elseif i==3
         temp=load('winequality-white.mat');
         data = temp.winequality_white;
    elseif i==4
         temp=load('PM10.mat');
         data = temp.PM10;
       elseif i==5
         temp=load('Gasturbine.mat');
         data = temp.Gasturbine;
    elseif i==6
         data=load('housing.data');
    elseif i==7
         data=load('airfoil_self_noise.dat');
    elseif i==8
         data=load('abalone.data');
    elseif i==9
         temp=load('Concrete_Data.mat');
         data = temp.Concrete_Data;
    else
         temp=load('CASP-Protein.mat');
         data = temp.CASP;
    end
X=data(:,1:end-1); y=data(:,end); y=y-mean(y);
X = zscore(X); [N0,M]=size(X);
N=round(N0*.7);
idsTrain=datasample(1:N0,N,'replace',false);
XTrain=X(idsTrain,:); yTrain=y(idsTrain);
XTest=X; XTest(idsTrain,:)=[];
yTest=y; yTest(idsTrain)=[];
Nbs=N; % batch size
% Specify the total number of rules; use the original features without dimensionality reduction
nRules=30; % number of rules, used in MBGD_RDA2 and MBGD_RDA2_T
% maxFeatures=5; % maximum number of features to use
% if M>maxFeatures
%     [~,XPCA,latent]=pca(X);
%     realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');
%     usedDim=min(maxFeatures,realDim98);
%     X=XPCA(:,1:usedDim); [N0,M]=size(X);
% end
% XTrain=X(idsTrain,:); XTest=X; XTest(idsTrain,:)=[];

for kk = 1:1
    alpha1 = LAMBDA(kk);
for ll = 1:length(LAMBDA)
    beta = LAMBDA(ll);
for ii = 1:length(LAMBDA)
    lambda1 = LAMBDA(ii);
fprintf('i=%d,kk=%d,ll=%d,ii=%d\n',i,kk,ll,ii);
[RMSEtrain(kk,ll,ii),RMSEtest(kk,ll,ii)]=OTSKparameter(XTrain,yTrain,XTest,yTest,alpha1,beta,lambda1,lambda2,P,nMFs,nIt);

end
end
end
filen = sprintf('result_%d',i);
save(filen,'RMSEtrain','RMSEtest')
end