function allhx= DNFC(x1,x2,SrcTag,options)
% allhx: (layers*d)xn stacked hidden representations
l1 =size(x1,2); % 
layers = options.layers;
lambda = options.lambda;
% noises = options.noises;

% noises= 0.9;%% a noise set can be used
noises=[0.5,0.6,0.7,0.8,0.9];
Accs=zeros(length(noises), 1); % 5*1 
HxSet = {};
for iter = 1:length(noises)
    options.noises = noises(iter);
    disp('stacking hidden layers...')
    allhx = [];
    for layer = 1:layers

        disp(['layer:',num2str(layer)]);
        newhx = NFC(x1,x2,options,'lin');
%         hx1 = NFC(x1,x2,options,'lin');
%         S = hx1(:,1:l1);
%         Acc1 = nnCroVal(S,SrcTag,5); 
%         hx2 = NFC(x1,x2,options,'rbf');
%         S = hx2(:,1:l1);
%         Acc2 = nnCroVal(S,SrcTag,5);
% 
%         if Acc1>Acc2
%             newhx = hx1;
%             disp('lin');
%         else
%             newhx = hx2;
%             disp('rbf');
%         end 
        allhx = [allhx; newhx];
        x1 = newhx(:,1:l1);%取前1979列
        x2 = newhx(:,l1+1:end); %取后面的
    end
    HxSet{iter} = allhx;
    S = allhx(:,1:l1);
    Acc = nnCroVal(S,SrcTag,5);
%     predicted_Label = cvKnn(S, S, SrcTag, 1);        
%     r=find(predicted_Label==SrcTag);
%     Acc = length(r)/length(SrcTag)*100;   
    Accs(iter, 1) = Acc;
end
[~, noiseIdx]=max(Accs);
BestNoise = noises(noiseIdx);
allhx = HxSet{noiseIdx};
 
 



