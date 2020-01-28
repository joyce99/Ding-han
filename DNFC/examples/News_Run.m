% an example using mSDA to generate features for sentiment analysis on the Amazon review dataset of (Blitzer et al., 2006), using only the top 5,000 features
% clear
% clc
clear all;
% benchmark = 'D:\publicjob\MATLAB\DNFC';
benchmark = pwd;
addpath(genpath(benchmark))

domains=cell(4,1);
domains{1}='books';
domains{2}='dvd';
domains{3}='electronics';
domains{4}='kitchen';

folds = 5;
% two hyper-parameters to be cross-validated
% number of mSDA layers to be stacked
layers=5;
% corruption level
noises=[0.5,0.6,0.7,0.8,0.9];

% read in the raw input
%load(benchmark,'\mSDA\examples\amazon.mat');
load('E:\代码\mSDA\examples\amazon.mat');
dimen = 5000;
xx = xx(1:dimen, :);%取前5000维

ACCs=zeros(length(noises), size(domains,1));%5行4列
Cs=zeros(length(noises), size(domains,1));%5行4列

% cross validate on the corruption level
for iter = 1:length(noises)
	noise=noises(iter);
	disp(['corruption level ', num2str(noise)])
	% generate hidden representations using mSDA
	[allhx] = mSDA(double(xx>0),noise,layers);
        for j = 1:size(domains,1)
		source=domains{j};
		disp(['domain ',source, ' ...'])
		yr=yy(offset(j)+1:offset(j)+2000);%取前2000个特征的标签
		xr=[xx(:, offset(j)+1:offset(j)+2000); allhx(:, offset(j)+1:offset(j)+2000)];
		xr=xr';
		Cs(iter, j) = 1./mean(sum(xr.*xr,2));
		model = svmtrain(yr,xr,['-q -t 0 -c ',num2str(Cs(iter,j)),' -m 3000']);
		ACCs(iter, j) = svmtrain(yr,xr,['-q -t 0 -c ',num2str(Cs(iter,j)),' -v ', num2str(folds), ' -m 3000']);
        end
	fprintf('\n')
end

% finalize training and testing
[temp, noiseIdx]=max(ACCs);
for j = 1:size(domains,1)
	source=domains{j};
	yr=yy(offset(j)+1:offset(j)+2000);
	bestNoise = noises(noiseIdx(j));
	disp(['learn representation with corruption level ' num2str(bestNoise), ' ...']);
	[allhx] = mSDA(double(xx>0), bestNoise, layers);
	xr=[xx(:, offset(j)+1:offset(j)+2000); allhx(:, offset(j)+1:offset(j)+2000)];
	xr=xr';
	bestC=Cs(noiseIdx(j),j);
	disp(['final training on domain ', source, ' ...'])
	model = svmtrain(yr,xr,['-q -t 0 -c ',num2str(bestC),' -m 3000']);
	for i = 1:size(domains,1)
		target=domains{i};
		if i == j
			continue;
		end
		disp(['final testing on domain ', target, ' ...'])
		xe=[xx(:, offset(i)+2001:offset(i+1)); allhx(:, offset(i)+2001:offset(i+1))];
		xe=xe';
		ye=yy(offset(i)+2001:offset(i+1));
		[label,accuracy] = svmpredict(ye,xe,model);
	end
	fprintf('\n');
end

%% prepare path
NN_Neighbours =1;
Datapath1= [benchmark,'\Data\Comp_vs_Rec.mat'];
%Comp_vs_Rec
% Comp_vs_Sci
% Comp_vs_Talk
load(Datapath1);
S_H = src_X';
SrcTag = src_labels;
T_H = tar_X';
TarTag = tar_labels;
SrcSamp = size(S_H,1);
S_H = double(S_H>0);
T_H = double(T_H>0);


             
% %% DNFC 
Result = [];
options.layers = 1;
options.lambda =1000;
options.beta =  0.001 ;
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
    Result =[Result;accuracy(1)]
        
                     







