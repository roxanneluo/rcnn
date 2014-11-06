function n = singular_num(data)
n = sum(sum(isnan(data) | isinf(data)));
