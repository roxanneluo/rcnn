function draw_class_sum(classes, num)
% num: num of class to draw in each class
VOCdevkit = './datasets/VOCdevkit2007';
opts = struct;
opts.net_file = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k'
opts.net_def_file = './model-defs/pascal_top_entropy_easy.prototxt';
opts.cache_name = 'v1_finetune_voc_2007_trainval_iter_70k';

cell_feat_opts = create_feat_opts(1, 5);
feat_opts = cell_feat_opts{1};
%layer = 5;
%feat_opts = struct('layer', layer, 'd', false, ...
 %       'w', w_or_r, 'combine', @l2, 'combine_name', 'l2');
global my_test_feat_opts
my_test_feat_opts = feat_opts;
sprintf(feat_opts_to_string(my_test_feat_opts))

imdb_train = imdb_from_voc(VOCdevkit, 'trainval', '2007');
%imdb_test = imdb_from_voc(VOCdevkit, 'test', '2007');
imdb = imdb_train;

conf = rcnn_config('sub_dir', imdb.name);

rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file,...
    opts.cache_name, true);
rcnn_model = rcnn_load_model(rcnn_model, conf.use_gpu);
rcnn_model.feat_opts = feat_opts;
opts.feat_norm_mean = rcnn_feature_stats(imdb, 7, rcnn_model);

save_file = sprintf('./feat_cache/%s/%s/gt_pos_layer_5_cache.mat', ...
    opts.cache_name, imdb.name)
try
  load(save_file);
  fprintf('Loaded saved positives from ground truth boxes\n');
catch
  fprintf('calculate positive features\n');
  [X_pos, keys_pos] = get_positive_pool5_features(imdb, opts);
  save(save_file, 'X_pos', 'keys_pos', '-v7.3');
end

fprintf('mkdirs\n');
% X_pos{i} => X_pos
feat_name = feat_opts_to_string(feat_opts);
draw_res = './draw-res/';
sum_folder = [draw_res, feat_name, '/summary/'];
all_folder = [draw_res, feat_name, '/all/'];

system(['mkdir -p ', draw_res]);
system(['mkdir -p ', draw_res, feat_name]);
system(['mkdir -p ', sum_folder]);
system(['mkdir -p ', all_folder]);
fprintf('mkdir done\n');

if ~exist('classes', 'var')
  classes = imdb.class_ids;
end
fprintf('draw classes:\n');
classes
for i = classes 
  fprintf('%14s has %6d positive instances\n', ...
      imdb.classes{i}, size(X_pos{i},1));
  pos = get_feature(X_pos{i}, rcnn_model);
  fprintf('after get_feature\n');
  pos = rcnn_scale_features(pos, opts.feat_norm_mean);
  fprintf('after scale\n');
  mean_pos = mean(pos, 1);
  stdd = std(pos);
  fprintf('after calculating mean and std\n');
  draw_summary(i, imdb.classes{i}, mean_pos, stdd, feat_name, sum_folder);
  print_summary(i, imdb.classes{i}, mean_pos, stdd, feat_name, sum_folder);
  fprintf('after save summary');
  %draw_all(i, imdb.classes{i}, X_pos, feat_name, sum_folder);
  fprintf('after save all\n');
end

% ------------------------------------------------------------------------
function print_summary(class_id, class, mean_pos, stdd, feat_name, folder)
% ------------------------------------------------------------------------
file_name = [folder, sprintf('%s-%d %s-summary', class, class_id, feat_name),'.csv']
f = fopen(file_name, 'w');
len = length(mean_pos);
print(f, 1:len);
print(f, mean_pos);
print(f, stdd);
fclose(f);

% ------------------------------------------------------------------------
function print(file, x)
% ------------------------------------------------------------------------
len = length(x);
for i=1:len
  fprintf(file, '%f, ', x(i));
end
fprintf(file, '\n');



% ------------------------------------------------------------------------
function draw_summary(class_id, class, mean_pos, stdd, feat_name, folder)
% ------------------------------------------------------------------------
figure();
title_str = sprintf('%s-%d %s-summary', class, class_id, feat_name);
title(title_str); hold on
%bar(mean_pos); hold on
%errorbar(mean_pos, stdd, 'linestyle', 'none');
errorbar(mean_pos, stdd);
saveas(gcf, [folder, title_str, '.jpg']);


% ------------------------------------------------------------------------
function draw_all(class_id, class, X_pos, feat_name, folder)
% ------------------------------------------------------------------------
figure();
title_str = sprintf('%s-%d %s-all', class, class_id, feat_name);
title(title_str); hold on;
fprintf('before surf\n');
surf(X_pos); hold on;
fprintf('after surf\n');
xlabel('num');
ylabel('feature dim');
fprintf('after label\n')
[folder, title_str, '.jpg']
saveas(gcf, [folder, title_str, '.jpg']);
fprintf('after saveas');


% ------------------------------------------------------------------------
function draw_lines(y)
% ------------------------------------------------------------------------
len = length(y);
x = zeros(1, 2*len);
yy = zeros(1, 2*len);
for i=1:len
  x(2*i-1) = i;
  x(2*i) = i;
  yy(2*i-1) = 0;
  yy(2*i) = y;
end



% ------------------------------------------------------------------------
function [X_pos, keys] = get_positive_pool5_features(imdb, opts)
% ------------------------------------------------------------------------
X_pos = cell(max(imdb.class_ids), 1);
keys = cell(max(imdb.class_ids), 1);

for i = 1:length(imdb.image_ids)
  tic_toc_print('%s: pos features %d/%d\n', ...
                procid(), i, length(imdb.image_ids));

  d = rcnn_load_cached_pool5_features(opts.cache_name, ...
      imdb.name, imdb.image_ids{i});

  for j = imdb.class_ids
    if isempty(X_pos{j})
      X_pos{j} = single([]);
      keys{j} = [];
    end
    sel = find(d.class == j);
    if ~isempty(sel)
      X_pos{j} = cat(1, X_pos{j}, d.feat(sel,:));
      keys{j} = cat(1, keys{j}, [i*ones(length(sel),1) sel]);
    end
  end
end

% combine fun
% input: actually a 3-dim [combine_along_dim, combine_across_dim, num]
% output: a combined vector along dim 2 of size [combine_across_dim, num]
% -----------------------------------------------------------------------------
function feat = l2(diff)
% -----------------------------------------------------------------------------
feat = sqrt(sum(diff.*diff, 1));

% -----------------------------------------------------------------------------
function feat = l1(diff)
% -----------------------------------------------------------------------------
feat = sum(abs(diff), 1);

% -----------------------------------------------------------------------------
function feat = max_abs(diff)
% -----------------------------------------------------------------------------
feat = max(abs(diff), 1);

% -----------------------------------------------------------------------------
function feat = max_pool(diff)
% -----------------------------------------------------------------------------
feat = max(diff, 1);

