% an example using mSDA to generate features for sentiment analysis on the Amazon review dataset of (Blitzer et al., 2006), using only the top 5,000 features
clear all;
benchmark = pwd;
addpath(genpath(benchmark))
% prepare path
NN_Neighbours =1;
Datapath1= [benchmark,'\Data\amazon.mat'];
load(Datapath1);

%Amazon datasets
domains=cell(4,1);
domains{1}='books';
domains{2}='dvd';
domains{3}='electronics';
domains{4}='kitchen';


% parameters dimen:the dimen of the data selected
% 取数据的前5000维特征
dimen =5000;
Result =[];

noise = 0.8;
layers = 1;

for first =1:1
    for second =1:2
        if first==second
            continue
        end
        disp(['source-->target：',domains{first},'-->',domains{second}]);
        % src_X：源领域数据
        % src_labels：源领域标签
        % tar_X：目标数据
        % tar_labels：目标数据标签
        src_X = xx(1:dimen, offset(first)+1:offset(first)+2000);
        src_labels = yy(offset(first)+1:offset(first)+2000);
        tar_X=xx(1:dimen, offset(second)+2001:offset(second+1)); 
        tar_labels=yy(offset(second)+2001:offset(second+1));
        
        [src_X,tar_X] = ZeroFeatureDelete(src_X,tar_X);
        
        S_H = src_X';
        SrcTag = src_labels;
        T_H = tar_X';
        TarTag = tar_labels;
        SrcSamp = size(S_H,1);
        S_H = double(S_H>0);
        T_H = double(T_H>0);
        
        options.layers = 1;
        options.lambda = 1000;
        options.beta = 0.001;
        % options.noises = 0.9;
        options.size = size(S_H,1);
        hx_dnfc = DNFC(S_H',T_H',SrcTag,options);
        S_dnfc = hx_dnfc(:,1:SrcSamp);
        T_dnfc = hx_dnfc(:,SrcSamp+1:end);
%         S_dnfc =[S_H'; hx_dnfc(:,1:SrcSamp)];
%         T_dnfc =[T_H'; hx_dnfc(:,SrcSamp+1:end)];
        xr=S_dnfc;
        xr=xr';
        bestC = 1./mean(sum(xr.*xr,2));
        model = svmtrain(SrcTag,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
        xe= T_dnfc;
        xe=xe';
        [label,accuracy] = svmpredict(TarTag,xe,model);
        Result =[Result;accuracy(1)]
    end
end

Result
xlswrite('F:\MATLAB\MDAGraphRegularized\Amazon_Result.xlsx',Result,'data2');

