
function classIX = get_base_IX(data)
%------------------------------------------------------
dim_mean = mean(data);
[~, classIX] = sort(dim_mean);
