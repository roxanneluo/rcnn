% dump non-normalized features
function feat_dim = dump_pos(feat_name, class_ids, data_dir, opts)
max_check = get_max_check(class_ids, opts);
[class_ids, max_check] = remove_exceed(class_ids, max_check, data_dir);

num_class = length(class_ids);
exceed = zeros(num_class,1);
feats = cell(num_class, 1);
feat_dim = 0;

matfiles = get_matfiles(class_ids, data_dir);
nums = get_dumped_nums(matfiles);
dump_log_filename = [data_dir 'dump.log'];
dump_log = fopen(dump_log_filename, 'a');

VOCdevkit = './datasets/VOCdevkit2007';
imdb = imdb_from_voc(VOCdevkit, 'trainval', '2007');

im_ids = imdb.image_ids;
for i = opts.im_start:length(im_ids)
  if all(exceed)
    break; 
  end

  tic_toc_print('%s: features from %s, %d/%d\n', ...
      procid(), im_ids{i}, i, length(im_ids));
  d = rcnn_load_cached_pool5_features(feat_name, imdb.name, ...
      im_ids{i}, true);
  feat_dim = size(d.feat,2);

  for j = 1:num_class
    if exceed(j)
      continue;
    end

    id = class_ids(j);
    sel = find(d.class == id);
    if ~isempty(sel)
      feats{j} = cat(1, feats{j}, d.feat(sel,:));
      nums(j) = nums(j) + length(sel);
      if nums(j) >= max_check(j)
        fprintf('%s exceeds max_check %d\n', ...
            imdb.classes{id}, max_check(j));
        exceed(j) = true;
        feats{j}(max_check(j)+1:end,:) = [] ;
        nums(j) = max_check(j);
      end
    end
  end
  if num_sum(feats) >= opts.dump_interval
    dump_and_log(feats, matfiles, i, im_ids{i}, dump_log, feat_dim, nums);
    for j = 1:num_class
      feats{j} = [];
    end
  end
end
% NOTE: i
dump_and_log(feats, matfiles, i, im_ids{i}, dump_log, feat_dim, nums);
assert(all(exceed == (nums>=max_check)));
non_exceed_err_log(nums, max_check, class_ids, data_dir); % if all exceed, return
fclose(dump_log);

if feat_dim <= 0
  d = rcnn_load_cached_pool5_features(feat_name, imdb.name, ...
      im_ids{1}, true);
  feat_dim = size(d.feat, 2);
end

%------------------------------------------------------------------------------
function non_exceed_err_log(nums, max_check, class_ids, data_dir)
%------------------------------------------------------------------------------
if all(nums >= max_check)
  return;
end

err_filename = [data_dir 'max_check_not_met.log']; 
f = fopen(err_filename, 'a');
for i = 1:num_class
  if nums(i) < max_check(i);
    id = class_ids(i);
    fprintf('size of class %s-%d is %d vs. max_check = %d\n', ...
        imdb.classes{id}, id, size(feats{i},1), max_check(i));
    fprintf(f, 'size of class %s-%d is %d vs. max_check = %d\n', ...
        imdb.classes{id}, id, size(feats{i},1), max_check(i));
  end
end
fclose(f);


%------------------------------------------------------------------------------
function nsum = num_sum(feats)
%------------------------------------------------------------------------------
nsum = 0;
for i = 1:length(feats)
  nsum = nsum + size(feats{i},1);
end

%------------------------------------------------------------------------------
function log_dump(action, im_id, im_name, nums, dump_log)
%------------------------------------------------------------------------------
log_str = sprintf('%s dumping image %d @ %s\n\t%s\n', action, im_id, im_name,...
    array_to_str(nums));
fprintf(log_str);
fprintf(dump_log, log_str);

%------------------------------------------------------------------------------
function arr_str = array_to_str(arr) 
%------------------------------------------------------------------------------
arr_str = '';
for i = 1:length(arr)
  arr_str = [arr_str int2str(arr(i)) ',' ];
end

%------------------------------------------------------------------------------
function dump_and_log(feats, matfiles, im_id, im_name, dump_log, feat_dim, nums)
%------------------------------------------------------------------------------
if num_sum(feats) == 0
  return;
end

assert(length(feats) == length(matfiles));
assert(feat_dim > 0);

log_dump('Start', im_id, im_name, nums, dump_log);

for i = 1:length(matfiles)
  try
    disp(size(matfiles{i}, 'data'));
  catch
    fprintf('%d has no data\n', i);
    matfiles{i}.data = single(zeros(0, feat_dim));
  end
  num = size(feats{i},1);
  if num > 0
    matfiles{i}.data(end+1:end+num,:) = ...
      feats{i};
    fprintf('after dumping %d: \t', i);
    disp(size(matfiles{i}, 'data'))
  end
end

log_dump('Finished', im_id, im_name, nums, dump_log);

%------------------------------------------------------------------------------
function nums = get_dumped_nums(matfiles)
%------------------------------------------------------------------------------
num_class = length(matfiles);
nums = zeros(num_class,1);
for i =1:num_class
  try
  [nums(i), ~] = size(matfiles{i}, 'data');
  catch
    nums(i) = 0;
  end
end

%------------------------------------------------------------------------------
function matfiles = get_matfiles(class_ids, dir)
%------------------------------------------------------------------------------
num_class = length(class_ids);
matfiles = cell(num_class,1);
for i = 1:num_class
  filename = get_data_filename(dir, class_ids(i), false);
  if ~exist(filename)
    save(filename, '-v7.3');
  end
  matfiles{i} = ...
    matfile(filename,'writable', true);
end


%------------------------------------------------------------------------------
function [class_ids,max_check] = remove_exceed(class_ids, max_check, dir)
%------------------------------------------------------------------------------
IX = [];
for i = 1:length(class_ids)
  id = class_ids(i);
  filename = get_data_filename(dir, id, false);
  if exist(filename, 'file');
    m = matfile(filename);
    try
      [num, ~] = size(m, 'data');
    catch
      num = 0;
    end
    disp(num)
    if num >= max_check(i)
      fprintf('%s exists and exceeds max_check %d\n', filename, max_check(i));
      IX = [IX, i];
    end
  end
end
class_ids(IX) = [];
max_check(IX) = [];
disp(class_ids)
disp(max_check');

%------------------------------------------------------------------------------
function max_check = get_max_check(class_ids, opts)
%------------------------------------------------------------------------------
max_check = load('num_pos.txt');
max_check = min(max_check(class_ids), opts.max_num_per_class);
fprintf('max_check:\n');
