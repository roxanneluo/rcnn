function dump_normalize(data_dir, num_class)
parfor i=1:num_class
  dump_norm_file(i, data_dir);
end

function dump_norm_file(class_id, data_dir)
ori_filename = get_data_filename(data_dir, class_id, false);
filename = get_data_filename(data_dir, class_id, true);
if ~exist(filename)
  fprintf('Loading %s\n', ori_filename);
  ld = load(ori_filename); 
  ld.data = normalize(ld.data);
  data = ld.data; clear ld;
  fprintf('Saving %s\n', filename);
  save(filename, 'data', '-v7.3');
  fprintf('Saved %s\n', filename);
end
