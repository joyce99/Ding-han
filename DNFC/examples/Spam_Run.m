% an example using mSDA to generate features for sentiment analysis on the Amazon review dataset of (Blitzer et al., 2006), using only the top 5,000 features
clear all;
benchmark = pwd;
addpath(genpath(benchmark))
%% prepare path
NN_Neighbours =1;
Datapath1= [benchmark,'\Data\spam_task_a_all.mat'];
load(Datapath1);
% parameters dimen:the dimen of the data selected
% 取数据的前5000维特征
dimen = 5000;

src_X = xx(1:dimen, 1:4000);
src_labels = yy(1:4000);
% tar_X存放目标数据
% tar_labels存放目标数据标签



Result =[];
for i = 1:3
    S_H = src_X';
    SrcTag = src_labels;
    tar_X=xx(1:dimen, 4001+(i-1)*2500:4000+i*2500); 
    tar_labels=yy( 4001+(i-1)*2500:4000+i*2500);
%     [target_data,target_label]=Remove_Sample(target_data,target_label);
    
    T_H = tar_X';
    TarTag = tar_labels;
    SrcSamp = size(S_H,1);
    S_H = double(S_H>0);
    T_H = double(T_H>0);

    options.layers = 1;
    options.lambda = 1 ;
    options.beta = 10;
    options.size = size(S_H,1);
    hx_dnfc = DNFC(S_H',T_H',SrcTag,options);
    
    S_dnfc = hx_dnfc(:,1:SrcSamp);
    T_dnfc = hx_dnfc(:,SrcSamp+1:end);
%     S_dnfc =[S_H'; hx_dnfc(:,1:SrcSamp)];
%     T_dnfc =[T_H'; hx_dnfc(:,SrcSamp+1:end)];
    xr=S_dnfc;
    xr=xr';
    bestC = 1./mean(sum(xr.*xr,2));
    model = svmtrain(SrcTag,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
    xe= T_dnfc;
    xe=xe';
    [label,accuracy] = svmpredict(TarTag,xe,model);
    accuracy(1)
%     S_dnfc = hx_dnfc(:,1:SrcSamp);
%     T_dnfc = hx_dnfc(:,SrcSamp+1:end);
%     % S_dnfc =[S_H'; hx_dnfc(:,1:SrcSamp)];
%     % T_dnfc =[T_H'; hx_dnfc(:,SrcSamp+1:end)];
%     predicted_Label = cvKnn(T_dnfc, S_dnfc, SrcTag, NN_Neighbours);        
%     r=find(predicted_Label==TarTag);
%     Acc_dnfc = length(r)/length(TarTag)*100   
end

% 
%  [allhx, Ws] = mSDA(double(total>0),corruption,layers);
%     xr=allhx(:,1:size(src_X,2));
%     xr=xr';
%     bestC = 1./mean(sum(xr.*xr,2));
%     model = svmtrain(src_labels,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
%     xe= allhx(:,size(src_X,2)+1:size(total,2));
%     xe=xe';
%     [label,accuracy] = svmpredict(target_label,xe,model);
%     Result =[Result;accuracy(1)];



