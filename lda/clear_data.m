function clear_data(feat_name, max_num_per_class, num_correct)
data_dir = ['./lda/data/' feat_name '/' int2str(max_num_per_class) '/'];
num_class = 20;
new_data_dir = [data_dir 'new/'];
mkdirs({new_data_dir});
parfor i = 1:num_class
  clear_class(i, new_data_dir, data_dir, num_correct(i));
end

function clear_class(class_id, new_data_dir, data_dir, num_correct)
in_filename = get_data_filename(data_dir, class_id, false);
filename = get_data_filename(new_data_dir, class_id, false);
assert(exist(in_filename, 'file') ~= 0);
fprintf('Loading %s\n', in_filename);
ld = load(in_filename, 'data');
data = ld.data; clear ld;
assert(size(data,1) >= num_correct);
data(num_correct+1:end,:) = [];
fprintf('\tSaving %s\n', filename);
save(filename, 'data', '-v7.3');
fprintf('\tSaved %s\n', filename);
