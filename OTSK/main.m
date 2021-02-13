%% Illustrate how MBGD_RDA, MBGD_RDA2, MBGD_RDA_T and MBGD_RDA2_T are used.
%% By Dongrui WU, drwu@hust.edu.cn

clc; clearvars; close all; %rng(0);

nMFs=2; % number of MFs in each input domain, used in MBGD_RDA and MBGD_RDA_T
LAMBDA = [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001];
% the important parameters for FTRL
alpha1 = LAMBDA(1);
beta = LAMBDA(3);
lambda1 = LAMBDA(5);
lambda2 = 1;
nIt1=2; % number of iterations
% Nbs1=N0; % batch size
% the important parameters for MBGD
alpha=.01; % initial learning rate
lambda=0.05; % L2 regularization coefficient
P=0.5; % DropRule rate
Nbs=1; % batch size

% temp=load('NO2.mat');
% data=temp.data;
% temp=load('winequality-red.mat'); 
% data = temp.winequality_red;
% temp=load('winequality-white.mat');
% data = temp.winequality_white;
% temp=load('PM10.mat');
% data = temp.PM10;
% data=load('housing.data');
% data=load('airfoil_self_noise.dat');
%  data=load('abalone.data');
% temp=load('Concrete_Data.mat');
% data = temp.Concrete_Data;
% temp=load('Gasturbine.mat');
% data = temp.Gasturbine;
temp=load('CASP-Protein.mat');
data = temp.CASP;
X=data(:,1:end-1); y=data(:,end); y=y-mean(y);
X = zscore(X); [N0,M]=size(X);
N=round(N0*.7);
nIt=nIt1*N; % number of iterations
idsTrain=datasample(1:N0,N,'replace',false);
XTrain=X(idsTrain,:); yTrain=y(idsTrain);
XTest=X; XTest(idsTrain,:)=[];
yTest=y; yTest(idsTrain)=[];
% RMSEtrain=zeros(1,nIt); RMSEtest=RMSEtrain;
% Specify the total number of rules; use the original features without dimensionality reduction
nRules=30; % number of rules, used in MBGD_RDA2 and MBGD_RDA2_T
% [RMSEtrain(1,:),RMSEtest(1,:)]=MBGD_RDA2(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Gaussian MFs
% [RMSEtrain(2,:),RMSEtest(2,:)]=MBGD_RDA2_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Trapezoidal MFs

% Dimensionality reduction, as in the following paper:
% Dongrui Wu, Ye Yuan, Jian Huang and Yihua Tan, "Optimize TSK Fuzzy Systems for Regression Problems: 
% Mini-Batch Gradient Descent with Regularization, DropRule and AdaBound (MBGD-RDA)," IEEE Trans. 
% on Fuzzy Systems, 2020, accepted.
% maxFeatures=5; % maximum number of features to use
% if M>maxFeatures
%     [~,XPCA,latent]=pca(X);
%     realDim98=find(cumsum(latent)>=.98*sum(latent),1,'first');
%     usedDim=min(maxFeatures,realDim98);
%     X=XPCA(:,1:usedDim); [N0,M]=size(X);
% end
% XTrain=X(idsTrain,:); XTest=X; XTest(idsTrain,:)=[];

% Specify the number of MFs in each input domain
% Assume x1 has two MFs X1_1 and X1_2; then, all rules involving the first FS of x1 use the same X1_1,
% and all rules involving the second FS of x1 use the same X1_2
for i = 1:15
fprintf('i=%d,MBGD-RDA\n',i);
[RMSEtrain(i,:),RMSEtest(i,:),t(i,:),C,Sigma]=MBGD_RDA(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nMFs,nIt,Nbs); % Gaussian MFs
% [RMSEtrain(2,:),RMSEtest(2,:),A,B,C,D]=MBGD_RDA_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nMFs,nIt,Nbs); % Trapezoidal MFs
fprintf('i=%d,OTSK\n',i);
[RMSEtrain1(i,:),RMSEtest1(i,:),t1(i,:)]=OTSK(XTrain,yTrain,XTest,yTest,alpha1,beta,lambda1,lambda2,P,nMFs,nIt1,C,Sigma);
end
% [RMSEtrain2,RMSEtest2]=OTSK1(XTrain,yTrain,XTest,yTest,alpha1,beta,lambda1,lambda2,P,nMFs,nIt1);
% % Specify the total number of rules; each rule uses different membership functions
% nRules=nMFs^M; % number of rules
% [RMSEtrain(5,:),RMSEtest(5,:)]=MBGD_RDA2(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Gaussian MFs
% [RMSEtrain(6,:),RMSEtest(6,:)]=MBGD_RDA2_T(XTrain,yTrain,XTest,yTest,alpha,lambda,P,nRules,nIt,Nbs); % Trapezoidal MFs

%% Plot results
% figure('Position', get(0, 'Screensize')); hold on;
% % linestyles={'k--','k-','g--','g-','b--','b-','r--','r-','m--','m-','c--','c-'};
% linestyles={'k--','k-','g--','g-'};
% for i=1:size(RMSEtrain,1)
%     plot(RMSEtrain(i,:),linestyles{2*i-1},'linewidth',1);
%     plot(RMSEtest(i,:),linestyles{2*i},'linewidth',2);
% end
% plot(t,RMSEtest,'r-','linewidth',2);hold on;
% plot(t1,RMSEtest1,'b-','linewidth',2);
figure;
shadedErrorBar(mean(t), RMSEtest, {@mean,@std}, 'lineprops', '-r')
hold on;
shadedErrorBar(mean(t1), RMSEtest1, {@mean,@std}, 'lineprops', '-b')
set(gca,'FontSize',12)  %是设置刻度字体大小
legend( 'MBGD-RDA test RMSE','OSTK test RMSE','FontSize',12)
xlabel('Time t(s)','FontSize',14,'FontWeight','bold'); ylabel('RMSE','FontSize',14,'FontWeight','bold'); axis tight;

% legend('Training RMSE1, RDA2 Gaussian','Test RMSE1, RDA2 Gaussian',...
%     'Training RMSE2, RDA2 Trapezoidal','Test RMSE2, RDA2 Trapezoidal',...
%     'Training RMSE3, RDA Gaussian','Test RMSE3, RDA Gaussian',...
%     'Training RMSE4, RDA Trapezoidal','Test RMSE4, RDA Trapezoidal',...
%     'Training RMSE5, RDA2 Gaussian','Test RMSE5, RDA2 Gaussian',...
%     'Training RMSE6, RDA2 Trapezoidal','Test RMSE6, RDA2 Trapezoidal',...
%     'location','eastoutside');
% legend('Training RMSE3, RDA Gaussian','Test RMSE3, RDA Gaussian',...
%     'Training RMSE4, RDA Trapezoidal','Test RMSE4, RDA Trapezoidal',...
%     'Training RMSE5, RDA2 Gaussian','Test RMSE5, RDA2 Gaussian',...
%     'location','eastoutside');


%% Plot the trapezoidal MFs obtained from MBGD_RDA_T
% figure('Position', get(0, 'Screensize'));
% for m=1:M
%     subplot(M,1,m); hold on;
%     for n=1:nMFs
%         plot([A(m,n) B(m,n) C(m,n) D(m,n)],[0 1 1 0],'linewidth',2);
%     end
% end
