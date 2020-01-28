% an example using mSDA to generate features for sentiment analysis on the Amazon review dataset of (Blitzer et al., 2006), using only the top 5,000 features
% clear
clear all;
% clc
% benchmark = 'D:\publicjob\MATLAB\DNFC';
benchmark = pwd;
addpath(genpath(benchmark))
%% prepare path
NN_Neighbours =1;

Datapath1= [benchmark,'\Data\PeoplePlaces.mat'];
% OrgsPeople
% OrgsPlaces
% PeoplePlaces 
load(Datapath1);
% S_H = DataStructure.DataMatrix;

S_H = dataset.src_X';% 1237 * 4771 
SrcTag = dataset.src_labels; % 1237 * 1


T_H = dataset.tar_X'; % 1208 * 4771
TarTag = dataset.tar_labels; % 1208 *1

SrcSamp = size(S_H,1); %1237

% S_H = double(S_H>0);
% T_H = double(T_H>0);

% total = [S_H;T_H]';
% [total] = tfidf(total);
% total = total';
% S_H = total(1:SrcSamp,:);
% T_H = total(SrcSamp+1:end,:);
% SrcTag =  DataStructure.LabelMatrix;
% % DataPath2= [benchmark,'\Data\Sci\DataMatrix.mat'];
% load(DataPath2);
% T_H = DataStructure.DataMatrix;
% TarTag = DataStructure.LabelMatrix;
% Data = [S_H;T_H];
% Data_norm = MaxMinNorm(Data);
% [n,d] = size(Data_norm);
% S_H = Data_norm(1:SrcSamp,:);
% T_H = Data_norm(SrcSamp+1:end,:);
% Data = [S_H;T_H];
% Data_norm = MaxMinNorm(Data);
% % Data_norm = Data;
% S_H = Data_norm(1:SrcSamp,:);
% T_H = Data_norm(SrcSamp+1:end,:);

%% mSDA
% hx_msda = mSDA(S_H',T_H',SrcTag,3);
% S_msda = hx_msda(:,1:SrcSamp);
% T_msda = hx_msda(:,SrcSamp+1:end);
% predicted_Label1 = cvKnn(T_msda, S_msda, SrcTag, NN_Neighbours);        
% r=find(predicted_Label1==TarTag);
% Acc_msda = length(r)/length(TarTag)*100
             
% %% DNFC 
options.layers = 1;
options.lambda = 0;
options.beta =   10;
% options.noises = 0.9;
options.size = size(S_H,1);
hx_dnfc = DNFC(S_H',T_H',SrcTag,options);

S_dnfc = hx_dnfc(:,1:SrcSamp);
T_dnfc = hx_dnfc(:,SrcSamp+1:end);
% S_dnfc =[S_H'; hx_dnfc(:,1:SrcSamp)];
% T_dnfc =[T_H'; hx_dnfc(:,SrcSamp+1:end)];
xr=S_dnfc;
xr=xr';
bestC = 1./mean(sum(xr.*xr,2));
model = svmtrain(SrcTag,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
xe= T_dnfc;
xe=xe';
[label,accuracy] = svmpredict(TarTag,xe,model);
accuracy(1)


% predicted_Label = cvKnn(T_dnfc, S_dnfc, SrcTag, NN_Neighbours);        
% r=find(predicted_Label==TarTag);
% Acc_dnfc = length(r)/length(TarTag)*100    








