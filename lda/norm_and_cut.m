function nums = norm_and_cut(data_dir, num_class, filter_dir, num_filter, ...
    do_normalize, feat_dim, opts)
filter_files = get_file_handles(filter_dir, num_filter);
dim = feat_dim / num_filter;
for i = 2:num_class
  data_filename = get_data_filename(data_dir, i, false);
  data_file = fopen(data_filename, 'r');
  norm_data_filename = get_data_filename(data_dir, i, true);
  norm_data_file = fopen(norm_data_filename, 'w');
  fprintf('Start norm cut %d\n', i);
  nums(i) = 0;
  while true
    data = fread(data_file, [feat_dim, opts.dump_interval], 'single')';
    nums(i) = nums(i)+size(data,1);
    if isempty(data)
      break;
    end
    fprintf('\tnum = %d, loaded [%d,%d]\n', nums(i), size(data,1), size(data,2));
    data = normalize(data);
    fprintf('\twriting normalized data\n');
    fwrite(norm_data_file, data', 'single');
    parfor j = 1:num_filter
      fprintf('\twriting filter %d\n', j);
      fwrite(filter_files{j}, data(:, (j-1)*dim+1:j*dim)', 'single');
      fprintf('\twritten filter %d\n', j);
    end
  end
  fprintf('Finished norm cut %d\n', i);
  fclose(norm_data_file);
  fclose(data_file);
end
for i = 1:num_filter
  fclose(filter_files{i});
end
fprintf('nums:');
disp(nums);


function files = get_file_handles(filter_dir, num_filter)
files = cell(num_filter, 1);
for i = 1:num_filter
  filename = get_data_filename(filter_dir, i, true);
  files{i} = fopen(filename, 'w');
  assert(files{i} >= 0);
end
