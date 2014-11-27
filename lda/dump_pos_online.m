% dump non-normalized features
function feat_dim = dump_pos_online(feat_opt, class_ids, data_dir, opts)
feat_dim = get_feat_dims(feat_opt);

max_check = get_max_check(class_ids, opts);
[class_ids, max_check, nums] = remove_exceed(class_ids, max_check, data_dir, feat_dim);

num_class = length(class_ids);
if num_class == 0
  return;
end
exceed = zeros(num_class,1);
feats = cell(num_class, 1);

files = get_file_handles(class_ids, data_dir);
dump_log_filename = [data_dir 'dump.log'];
dump_log = fopen(dump_log_filename, 'a');

VOCdevkit = './datasets/VOCdevkit2007';
imdb = imdb_from_voc(VOCdevkit, 'test', '2007');

roidb = imdb.roidb_func(imdb);
rcnn_model = prepare_model(feat_opt, opts)

im_ids = imdb.image_ids;
for i = opts.im_start:length(im_ids)
  if all(exceed)
    break; 
  end

  tic_toc_print('%s: features from %s, %d/%d\n', ...
      procid(), im_ids{i}, i, length(im_ids));
  d = compute_grad(rcnn_model, feat_opt, i, imdb, roidb, opts);

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
        feats{j}(end-nums(j)+max_check(j)+1:end,:) = [] ;
        nums(j) = max_check(j);
      end
    end
  end
  
  if num_sum(feats) >= opts.dump_interval
    dump_and_log(feats, files, i, im_ids{i}, dump_log, feat_dim, nums);
    for j = 1:num_class
      feats{j} = [];
    end
  end
end
% NOTE: i
dump_and_log(feats, files, i, im_ids{i}, dump_log, feat_dim, nums);
disp(nums);
assert(all(exceed == (nums>=max_check)));
non_exceed_err_log(nums, max_check, class_ids, data_dir); % if all exceed, return
fclose(dump_log);

for i = 1:length(files)
  fclose(files{i});
end

%------------------------------------------------------------------------------
function non_exceed_err_log(nums, max_check, class_ids, data_dir)
%------------------------------------------------------------------------------
if all(nums >= max_check)
  return;
end

num_class = length(class_ids);
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
function dump_and_log(feats, files, im_id, im_name, dump_log, feat_dim, nums)
%------------------------------------------------------------------------------
if num_sum(feats) == 0
  return;
end

assert(length(feats) == length(files));
assert(feat_dim > 0);

log_dump('Start', im_id, im_name, nums, dump_log);

parfor i = 1:length(files)
  num = size(feats{i},1);
  if num > 0
    fwrite(files{i}, feats{i}', 'single');  
  end
end

log_dump('Finished', im_id, im_name, nums, dump_log);

%------------------------------------------------------------------------------
function nums = get_dumped_nums(files)
%------------------------------------------------------------------------------
num_class = length(files);
nums = zeros(num_class,1);
for i =1:num_class
  try
  [nums(i), ~] = size(files{i}, 'data');
  catch
    nums(i) = 0;
  end
end

%------------------------------------------------------------------------------
function files = get_file_handles(class_ids, dir)
%------------------------------------------------------------------------------
num_class = length(class_ids);
files = cell(num_class,1);
for i = 1:num_class
  filename = get_data_filename(dir, class_ids(i), false);
  files{i} = fopen(filename, 'a');
end


%------------------------------------------------------------------------------
function [class_ids,max_check, nums] = remove_exceed(class_ids, max_check, dir, feat_dim)
%------------------------------------------------------------------------------
exceed = false(length(class_ids));
nums = zeros(length(class_ids),1);
parfor i = 1:length(class_ids)
  nums(i) = get_num(dir, class_ids(i), feat_dim);
  exceed(i) = max_check(i) <= nums(i);
end
IX = find(exceed);
class_ids(IX) = [];
max_check(IX) = [];
nums(IX) = [];
disp(class_ids)
disp(max_check');
disp(nums');

%------------------------------------------------------------------------------
function num = get_num(dir, class_id, feat_dim)
%------------------------------------------------------------------------------
filename = get_data_filename(dir, class_id, false);
if ~exist(filename, 'file');
  num = 0;
else
  f = fopen(filename, 'r');
  data = fread(f, [feat_dim, inf], 'single');
  fclose(f);
  num = size(data, 2);
  clear data;
end


%------------------------------------------------------------------------------
function max_check = get_max_check(class_ids, opts)
%------------------------------------------------------------------------------
max_check = load('num_pos.txt');
max_check = min(max_check(class_ids), opts.max_num_per_class);
fprintf('max_check:\n');

%{

  %}
