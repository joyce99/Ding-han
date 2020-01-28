% clear
% clc
% benchmark = 'D:\publicjob\MATLAB\DNFC';
benchmark = pwd;
addpath(genpath(benchmark))
%% prepare path
NN_Neighbours =1;
% Datapath1= [benchmark,'\Data\Talk\DataMatrix.mat'];
Datapath1= [benchmark,'\Data\Talk\DataMatrix.mat'];

load(Datapath1);
S_H = DataStructure.DataMatrix;
SrcSamp = size(S_H,1);
SrcTag =  DataStructure.LabelMatrix;
DataPath2= [benchmark,'\Data\Sci\DataMatrix.mat'];
load(DataPath2);
T_H = DataStructure.DataMatrix;
TarTag = DataStructure.LabelMatrix;
Data = [S_H;T_H];
Data_norm = MaxMinNorm(Data);
[n,d] = size(Data_norm);
S_H = Data_norm(1:SrcSamp,:);
T_H = Data_norm(SrcSamp+1:end,:);
Data = [S_H;T_H];
Data_norm = MaxMinNorm(Data);
S_H = Data_norm(1:SrcSamp,:);
T_H = Data_norm(SrcSamp+1:end,:);

%% mSDA
% hx_msda = mSDA(S_H',T_H',SrcTag,3);
% S_msda = hx_msda(:,1:SrcSamp);
% T_msda = hx_msda(:,SrcSamp+1:end);
% predicted_Label1 = cvKnn(T_msda, S_msda, SrcTag, NN_Neighbours);        
% r=find(predicted_Label1==TarTag);
% Acc_msda = length(r)/length(TarTag)*100
             
% %% DNFC 
options.layers = 3;
options.lambda = 1000;
options.beta = 0;
% options.noises = 0.7;
options.size = size(S_H,1);
hx_dnfc = DNFC(S_H',T_H',SrcTag,options);
S_dnfc = hx_dnfc(:,1:SrcSamp);
T_dnfc = hx_dnfc(:,SrcSamp+1:end);
% S_dnfc =[S_H'; hx_dnfc(:,1:SrcSamp)];
% T_dnfc =[T_H'; hx_dnfc(:,SrcSamp+1:end)];
predicted_Label = cvKnn(T_dnfc, S_dnfc, SrcTag, NN_Neighbours);        
r=find(predicted_Label==TarTag);
Acc_dnfc = length(r)/length(TarTag)*100                        