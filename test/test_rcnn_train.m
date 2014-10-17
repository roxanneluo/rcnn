function [rcnn_model, rcnn_k_fold_model] = ...
    rcnn_train(imdb, varargin)
% [rcnn_model, rcnn_k_fold_model] = rcnn_train(imdb, varargin)
%   Trains an R-CNN detector for all classes in the imdb.
%   
%   Keys that can be passed in:
%
%   svm_C             SVM regularization parameter
%   bias_mult         Bias feature value (for liblinear)
%   pos_loss_weight   Cost factor on hinge loss for positives
%   layer             Feature layer to use (either 5, 6 or 7)
%   k_folds           Train on folds of the imdb
%   checkpoint        Save the rcnn_model every checkpoint images
%   crop_mode         Crop mode (either 'warp' or 'square')
%   crop_padding      Amount of padding in crop
%   net_file          Path to the Caffe CNN to use
%   cache_name        Path to the precomputed feature cache

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% TODO:
%  - allow training just a subset of classes

ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addParamValue('svm_C',           10^-3,  @isscalar);
ip.addParamValue('bias_mult',       10,     @isscalar);
ip.addParamValue('pos_loss_weight', 2,      @isscalar);
ip.addParamValue('layer',           7,      @isscalar);
ip.addParamValue('k_folds',         2,      @isscalar);
ip.addParamValue('checkpoint',      0,      @isscalar);
ip.addParamValue('crop_mode',       'warp', @isstr);
ip.addParamValue('crop_padding',    16,     @isscalar);
ip.addParamValue('net_file', ...
    './data/caffe_nets/finetune_voc_2007_trainval_iter_70k', ...
    @isstr);
ip.addParamValue('cache_name', ...
    'v1_finetune_voc_2007_trainval_iter_70000', @isstr);


ip.parse(imdb, varargin{:});
opts = ip.Results;

%opts.net_def_file = './model-defs/rcnn_batch_256_output_fc7.prototxt';
opts.net_def_file = './model-defs/pascal_top_entropy_easy.prototxt';

conf = rcnn_config('sub_dir', imdb.name);

% Record a log of the training and test procedure
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = [conf.cache_dir 'test_rcnn_train_' timestamp '.txt'];
%diary(diary_file);
fprintf('Logging output in %s\n', diary_file);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Training options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% ------------------------------------------------------------------------
% Create a new rcnn model
rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file,...
    opts.cache_name, true);
rcnn_model = rcnn_load_model(rcnn_model, conf.use_gpu);
rcnn_model.detectors.crop_mode = opts.crop_mode;
rcnn_model.detectors.crop_padding = opts.crop_padding;
rcnn_model.classes = imdb.classes;
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Get the average norm of the features
rcnn_model.feat_opts = conf.feat_opts;
[opts.feat_norm_mean, stdd] = rcnn_feature_stats(imdb, opts.layer, rcnn_model);
fprintf('average norm = %.3f, stdd = %f\n', opts.feat_norm_mean, stdd);



% ------------------------------------------------------------------------
function [X_neg, keys] = sample_negative_features(first_time, rcnn_model, ...
                                                  caches, imdb, ind)
% ------------------------------------------------------------------------
opts = rcnn_model.training_opts;

d = rcnn_load_cached_pool5_features(opts.cache_name, ...
    imdb.name, imdb.image_ids{ind});

class_ids = imdb.class_ids;

if isempty(d.feat)
  X_neg = cell(max(class_ids), 1);
  keys = cell(max(class_ids), 1);
  return;
end

%d.feat = rcnn_pool5_to_fcX(d.feat, opts.layer, rcnn_model);
%d.feat = rcnn_scale_features(d.feat, opts.feat_norm_mean);
d.feat = get_feature(d.feat, rcnn_model);
d.feat = rcnn_scale_features(d.feat, opts.feat_norm_mean);
sprintf('%d features in image %d', size(d.feat, 1), ind)

neg_ovr_thresh = 0.3;

if first_time
  for cls_id = class_ids
    I = find(d.overlap(:, cls_id) < neg_ovr_thresh);
    X_neg{cls_id} = d.feat(I,:);
    keys{cls_id} = [ind*ones(length(I),1) I];
  end
else
  zs = bsxfun(@plus, d.feat*rcnn_model.detectors.W, rcnn_model.detectors.B);
  for cls_id = class_ids
    z = zs(:, cls_id);
    I = find((z > caches{cls_id}.hard_thresh) & ...
             (d.overlap(:, cls_id) < neg_ovr_thresh));

    % Avoid adding duplicate features
    keys_ = [ind*ones(length(I),1) I];
    if ~isempty(caches{cls_id}.keys_neg) && ~isempty(keys_)
      [~, ~, dups] = intersect(caches{cls_id}.keys_neg, keys_, 'rows');
      keep = setdiff(1:size(keys_,1), dups);
      I = I(keep);
    end

    % Unique hard negatives
    X_neg{cls_id} = d.feat(I,:);
    keys{cls_id} = [ind*ones(length(I),1) I];
  end
end


% ------------------------------------------------------------------------
function [w, b] = update_model(cache, opts, pos_inds, neg_inds)
% ------------------------------------------------------------------------
solver = 'liblinear';
liblinear_type = 3;  % l2 regularized l1 hinge loss
%liblinear_type = 5; % l1 regularized l2 hinge loss

if ~exist('pos_inds', 'var') || isempty(pos_inds)
  num_pos = size(cache.X_pos, 1);
  pos_inds = 1:num_pos;
else
  num_pos = length(pos_inds);
  fprintf('[subset mode] using %d out of %d total positives\n', ...
      num_pos, size(cache.X_pos,1));
end
if ~exist('neg_inds', 'var') || isempty(neg_inds)
  num_neg = size(cache.X_neg, 1);
  neg_inds = 1:num_neg;
else
  num_neg = length(neg_inds);
  fprintf('[subset mode] using %d out of %d total negatives\n', ...
      num_neg, size(cache.X_neg,1));
end

switch solver
  case 'liblinear'
    ll_opts = sprintf('-w1 %.5f -c %.5f -s %d -B %.5f', ...
                      opts.pos_loss_weight, opts.svm_C, ...
                      liblinear_type, opts.bias_mult);
    fprintf('liblinear opts: %s\n', ll_opts);
    X = sparse(size(cache.X_pos,2), num_pos+num_neg);
    X(:,1:num_pos) = cache.X_pos(pos_inds,:)';
    X(:,num_pos+1:end) = cache.X_neg(neg_inds,:)';
    y = cat(1, ones(num_pos,1), -ones(num_neg,1));
    llm = liblinear_train(y, X, ll_opts, 'col');
    w = single(llm.w(1:end-1)');
    b = single(llm.w(end)*opts.bias_mult);

  otherwise
    error('unknown solver: %s', solver);
end


% ------------------------------------------------------------------------
function [W, B, folds] = update_model_k_fold(rcnn_model, caches, imdb)
% ------------------------------------------------------------------------
opts = rcnn_model.training_opts;
num_images = length(imdb.image_ids);
folds = create_folds(num_images, opts.k_folds);
W = cell(opts.k_folds, 1);
B = cell(opts.k_folds, 1);

fprintf('Training k-fold models\n');
for i = imdb.class_ids
  fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
  fprintf('Training folds for class %s\n', imdb.classes{i});
  fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');
  for f = 1:length(folds)
    fprintf('Holding out fold %d\n', f);
    [pos_inds, neg_inds] = get_cache_inds_from_fold(caches{i}, folds{f});
    [new_w, new_b] = update_model(caches{i}, opts, ...
        pos_inds, neg_inds);
    W{f}(:,i) = new_w;
    B{f}(i) = new_b;
  end
end


% ------------------------------------------------------------------------
function [pos_inds, neg_inds] = get_cache_inds_from_fold(cache, fold)
% ------------------------------------------------------------------------
pos_inds = find(ismember(cache.keys_pos(:,1), fold) == false);
neg_inds = find(ismember(cache.keys_neg(:,1), fold) == false);


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


% ------------------------------------------------------------------------
function cache = init_cache(X_pos, keys_pos)
% ------------------------------------------------------------------------
cache.X_pos = X_pos;
cache.X_neg = single([]);
cache.keys_neg = [];
cache.keys_pos = keys_pos;
cache.num_added = 0;
cache.retrain_limit = 2000;
cache.evict_thresh = -1.2;
cache.hard_thresh = -1.0001;
cache.pos_loss = [];
cache.neg_loss = [];
cache.reg_loss = [];
cache.tot_loss = [];
