function [data_new,label_new]=Remove_Sample(data,label)
% data:dxn
%label:nx1
    data_sum = sum(data,1);
    ind = find(data_sum==0);
    if(size(ind,2))
        for i=1:size(ind,2)
            data(:,ind(i))=[];
            label(ind(i))=[];
        end
    end
    data_new = data;
    label_new = label;
end
