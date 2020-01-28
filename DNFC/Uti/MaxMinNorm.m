function Output =  MaxMinNorm(Input)
D = size(Input,2);
for i =1:D
    Output(:,i) = (Input(:,i)-min(Input(:,i)))/(max(Input(:,i))-min(Input(:,i)));
end