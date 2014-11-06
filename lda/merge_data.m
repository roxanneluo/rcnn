function merge_data(dir1, dir2, new_dir)
num_class = 20;
parfor i=1:num_class
  merge_class_data(i, dir1, dir2, new_dir);
end

function merge_class_data(class_id, dir1, dir2, new_dir)
filename1 = get_data_filename(dir1, class_id, false);
filename2 = get_data_filename(dir2, class_id, false);
filename = get_data_filename(new_dir, class_id, false);
fprintf('Loading 1 %s\n', filename1);
ld1 = load(filename1, 'data');
fprintf('Loading 2 %s\n', filename2);
ld2 = load(filename2, 'data');
data = [ld1.data; ld2.data];
fprintf('Saving %s\n', filename);
save(filename, 'data', '-v7.3');
fprintf('Saved %s\n', filename);

