function Acc = DoCVKNN(Data,Tag,fold)
%%% data: d*n
DatL = size(Data,2);
N = floor(DatL/fold);
index = 1:DatL;
AccSet = [];
for i = 1:fold
    TestIndex = (i-1)*N+1:N*i;
    TrainIndex = setdiff(index,TestIndex);
    TrainDat = Data(:,TrainIndex);
    TestDat = Data(:,TestIndex);
    TrainTag = Tag(TrainIndex);
    TestTag = Tag(TestIndex);
    predicted_Label = cvKnn(TestDat, TrainDat, TrainTag, 1);
    r=find(predicted_Label==TestTag);
    accuracy = length(r)/length(TestTag)*100;
    AccSet = [AccSet,accuracy];
end
Acc = mean(AccSet);
