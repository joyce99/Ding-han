function allhx= mSDA(x1,x2,SrcTag,layers)
% allhx: (layers*d)xn stacked hidden representations
l1 =size(x1,2);
% noises=[0.5,0.6,0.7,0.8,0.9]; %% a noise set can be used
noises=0.9; %% a noise set can be used
ACCs=zeros(length(noises), 1);
HxSet = {};
for iter = 1:length(noises)
    noise=noises(iter);
    disp('stacking hidden layers...')
    prevhx = [x1,x2];
    allhx = [];
    for layer = 1:layers
        disp(['layer:',num2str(layer)])
        newhx = mDA(prevhx,noise);
        allhx = [allhx; newhx];
        prevhx = newhx;
    end
    HxSet{iter} = allhx;
    S = allhx(:,1:l1);
    Acc = nnCroVal(S,SrcTag,5);
    ACCs(iter, 1) = Acc;
end
[~, noiseIdx]=max(ACCs);
BestNoise = noises(noiseIdx);
allhx = HxSet{noiseIdx};

