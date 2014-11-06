function [data, nums] = load_dims(data_dir, dim_start, dim_end, num_cls,...
    do_normalize)
nums = zeros(num_cls, 1);
data_cell = cell(num_cls,1);
parfor i = 1:num_cls
  data_cell{i} = load_dims_of_class(i, dim_start, dim_end, data_dir, do_normalize);
end
data = [];
for i = 1:num_cls
  nums(i) = size(data_cell{i}, 1);
  data = cat(1, data, data_cell{i});
  data_cell{i} = [];
end

function data = load_dims_of_class(class_id, dim_start, dim_end, data_dir, do_normalize)
filename = get_data_filename(data_dir, class_id, do_normalize);
fprintf('Loading %s for dim %d:%d\n', filename, dim_start, dim_end);
m = matfile(filename);
data = m.data(:, dim_start:dim_end);
clear m;
fprintf('max: %f, absmean: %f, mean:%f\n', max(max(data)), mean(abs(data(:))), mean(data(:)));
