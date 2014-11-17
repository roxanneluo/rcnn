%------------------------------------------------------------------------------
function data_cos = get_cos(dots, data_norm, mean_norm) 
%------------------------------------------------------------------------------
norm_prod = data_norm*mean_norm;
norm_prod(norm_prod < eps) = 1;
data_cos = dots./norm_prod;
