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
ip.addParamValue('norm_weight',     true,       @isscalar);
ip.addParamValue('equal_dim',       true,       @isscalar);
ip.addParamValue('proj',            true,       @isscalar);
ip.addParamValue('whiten',          true,       @isscalar);
ip.addParamValue('pca_ratio',          1,       @isscalar);

ip.parse(imdb, varargin{:});
opts = ip.Results;
%topts.net_def_file = './model-defs/rcnn_batch_256_output_fc7.prototxt';
%opts.net_def_file = './model-defs/pascal_top_entropy_easy.prototxt';
%opts.net_def_file = './model-defs/pascal_top_softmax_back.prototxt';
opts.net_def_file = 'model-defs/nizf_batch_128_output_fc2.prototxt';
fprintf('===================opts.proj=%d, whiten=%d,pca_ratio=%f=================\n', opts.proj, opts.whiten, opts.pca_ratio);
conf = rcnn_config('sub_dir', imdb.name);

% Record a log of the training and test procedure
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = [conf.cache_dir 'rcnn_train_' timestamp '.txt'];
diary(diary_file);
fprintf('Logging output in %s\n', diary_file);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Training options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% ------------------------------------------------------------------------
% Create a new rcnn model
rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file,...
    conf.cache_dir);
rcnn_model = rcnn_load_model(rcnn_model, conf.use_gpu);
rcnn_model.detectors.crop_mode = opts.crop_mode;
rcnn_model.detectors.crop_padding = opts.crop_padding;
rcnn_model.classes = imdb.classes;
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Get feature dimensions
rcnn_model.feat_opts = conf.feat_opts;
rcnn_model.dims = get_feat_dims(rcnn_model.feat_opts);
rcnn_model.feat_dim = sum(rcnn_model.dims);
%rcnn_model.exist_r = exist_response(rcnn_model.feat_opts);
%rcnn_model.exist_w = exist_weight(rcnn_model.feat_opts);
rcnn_model.norm_weight = opts.norm_weight;
rcnn_model.equal_dim = opts.equal_dim;
rcnn_model.proj = opts.proj;
rcnn_model.whiten = opts.whiten;
rcnn_model.proj_dim = rcnn_model.feat_dim;

% ------------------------------------------------------------------------
% Get the average norm of the features
if opts.proj
  fprintf('load eigen\n');
  eigen = load_eigen(rcnn_model)
  if opts.pca_ratio < 1
    s = sum(eigen.eigen_val);
    p_sum = 0;
    for i = 1:rcnn_model.feat_dim
      p_sum = p_sum + eigen.eigen_val(i);
      if p_sum >= opts.pca_ratio*s
        rcnn_model.proj_dim = i;
        break;
      end
    end
    fprintf('~~~~~~~~~~~~~pcaRatio=%f\tproj_dim=%d~~~~~~~~~~~~~~~~~~~~~~~~\n', opts.pca_ratio, rcnn_model.proj_dim);
    eigen.eigen_vec = eigen.eigen_vec(:,1:rcnn_model.proj_dim);
    eigen.eigen_val = eigen.eigen_val(1:rcnn_model.proj_dim);
    rcnn_model.feat_dim = rcnn_model.proj_dim; %TODO
    rcnn_model.dims = rcnn_model.proj_dim;
  end
else
  eigen = [];
end
fprintf('rcnn_model.cache_name = %s\n', rcnn_model.cache_name);
[opts.feat_norm_mean, stdd] = rcnn_feature_stats(imdb, opts.layer, rcnn_model, eigen);
print_array('average norm', opts.feat_norm_mean);
print_array('std', stdd);
rcnn_model.training_opts = opts;
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Get all positive examples
% We cache only the pool5 features and convert them on-the-fly to
% fc6 or fc7 as required
rcnn_model.cache_name
save_file = sprintf('./feat_cache/%s/%s/gt_pos_cache.mat', ...
    rcnn_model.cache_name, imdb.name);
try
  load(save_file);
  fprintf('Loaded saved positives from ground truth boxes\nfile_name=%s\n', save_file);
catch
  [X_pos, keys_pos, im_IX] = get_positive_pool5_features(rcnn_model, imdb, opts, eigen);
  save(save_file, 'X_pos', 'keys_pos', 'im_IX', '-v7.3');
end
% Init training caches
caches = {};
for i = imdb.class_ids
  fprintf('%14s has %6d positive instances\n', ...
      imdb.classes{i}, size(keys_pos{i},1));
  %X_pos{i} = rcnn_pool5_to_fcX(X_pos{i}, opts.layer, rcnn_model);
  %X_pos{i} = rcnn_scale_features(X_pos{i}, opts.feat_norm_mean);
  %fprintf('~~~~~~~~~~class %d: len of pos = %d, num of im = %d~~~~~~~~~~~~\n',i, size(X_pos{i},1), size(im_IX(i), 1));
  %X_pos{i} = get_feature(rcnn_model, imdb.name, im_IX{i}, eigen);
  X_pos{i} = rcnn_scale_features(X_pos{i}, opts.feat_norm_mean, rcnn_model);
  fprintf('size of positive = [%d, %d]\n', size(X_pos{i},1), size(X_pos{i},2));
  caches{i} = init_cache(X_pos{i}, keys_pos{i});
end
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Train with hard negative mining
first_time = true;
% one pass over the data is enough
max_hard_epochs = 1;

for hard_epoch = 1:max_hard_epochs
  for i = 1:length(imdb.image_ids)
    fprintf('%s: hard neg epoch: %d/%d image: %d/%d\n', ...
            procid(), hard_epoch, max_hard_epochs, i, length(imdb.image_ids));

    % Get hard negatives for all classes at once (avoids loading feature cache
    % more than once)
    [X, keys] = sample_negative_features(first_time, rcnn_model, caches, ...
        imdb, i, eigen);
    % Add sampled negatives to each classes training cache, removing
    % duplicates
    for j = imdb.class_ids
      if ~isempty(keys{j})
        if ~isempty(caches{j}.keys_neg)
          [~, ~, dups] = intersect(caches{j}.keys_neg, keys{j}, 'rows');
          assert(isempty(dups));
        end
        caches{j}.X_neg = cat(1, caches{j}.X_neg, X{j});
        caches{j}.keys_neg = cat(1, caches{j}.keys_neg, keys{j});
        caches{j}.num_added = caches{j}.num_added + size(keys{j},1);
      end

      % Update model if
      %  - first time seeing negatives
      %  - more than retrain_limit negatives have been added
      %  - its the final image of the final epoch
      is_last_time = (hard_epoch == max_hard_epochs && i == length(imdb.image_ids));
      hit_retrain_limit = (caches{j}.num_added > caches{j}.retrain_limit);
      if (first_time || hit_retrain_limit || is_last_time) && ...
          ~isempty(caches{j}.X_neg)
        fprintf('>>> Updating %s detector <<<\n', imdb.classes{j});
        fprintf('Cache holds %d pos examples %d neg examples\n', ...
                size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
        [new_w, new_b] = update_model(caches{j}, opts);
        rcnn_model.detectors.W(:, j) = new_w;
        rcnn_model.detectors.B(j) = new_b;
        caches{j}.num_added = 0;

        z_pos = caches{j}.X_pos * new_w + new_b;
        z_neg = caches{j}.X_neg * new_w + new_b;

        caches{j}.pos_loss(end+1) = opts.svm_C * opts.pos_loss_weight * ...
                                    sum(max(0, 1 - z_pos));
        caches{j}.neg_loss(end+1) = opts.svm_C * sum(max(0, 1 + z_neg));
        caches{j}.reg_loss(end+1) = 0.5 * new_w' * new_w + ...
                                    0.5 * (new_b / opts.bias_mult)^2;
        caches{j}.tot_loss(end+1) = caches{j}.pos_loss(end) + ...
                                    caches{j}.neg_loss(end) + ...
                                    caches{j}.reg_loss(end);

        for t = 1:length(caches{j}.tot_loss)
          fprintf('    %2d: obj val: %.3f = %.3f (pos) + %.3f (neg) + %.3f (reg)\n', ...
                  t, caches{j}.tot_loss(t), caches{j}.pos_loss(t), ...
                  caches{j}.neg_loss(t), caches{j}.reg_loss(t));
        end

        % store negative support vectors for visualizing later
        SVs_neg = find(z_neg > -1 - eps);
        rcnn_model.SVs.keys_neg{j} = caches{j}.keys_neg(SVs_neg, :);
        rcnn_model.SVs.scores_neg{j} = z_neg(SVs_neg);

        % evict easy examples
        easy = find(z_neg < caches{j}.evict_thresh);
        caches{j}.X_neg(easy,:) = [];
        caches{j}.keys_neg(easy,:) = [];
        fprintf('  Pruning easy negatives\n');
        fprintf('  Cache holds %d pos examples %d neg examples\n', ...
                size(caches{j}.X_pos,1), size(caches{j}.X_neg,1));
        fprintf('  %d pos support vectors\n', numel(find(z_pos <  1 + eps)));
        fprintf('  %d neg support vectors\n', numel(find(z_neg > -1 - eps)));
      end
    end
    first_time = false;

    if opts.checkpoint > 0 && mod(i, opts.checkpoint) == 0
      save([conf.cache_dir 'rcnn_model'], 'rcnn_model');
    end
  end
end
% save the final rcnn_model
save([conf.cache_dir 'rcnn_model'], 'rcnn_model');
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
if opts.k_folds > 0
  rcnn_k_fold_model = rcnn_model;
  [W, B, folds] = update_model_k_fold(rcnn_model, caches, imdb);
  rcnn_k_fold_model.folds = folds;
  for f = 1:length(folds)
    rcnn_k_fold_model.detectors(f).W = W{f};
    rcnn_k_fold_model.detectors(f).B = B{f};
  end
  save([conf.cache_dir 'rcnn_k_fold_model'], 'rcnn_k_fold_model');
else
  rcnn_k_fold_model = [];
end
% ------------------------------------------------------------------------


% ------------------------------------------------------------------------
function [X_neg, keys] = sample_negative_features(first_time, rcnn_model, ...
                                                  caches, imdb, ind, eigen)
% ------------------------------------------------------------------------
opts = rcnn_model.training_opts;
%{
d = rcnn_load_cached_pool5_features(opts.cache_name, ...
    imdb.name, imdb.image_ids{ind});
%}

d = rcnn_load_cached_pool5_features(opts.cache_name, ...
    imdb.name, imdb.image_ids{ind}, false, {'overlap'});

class_ids = imdb.class_ids;

%d.feat = rcnn_pool5_to_fcX(d.feat, opts.layer, rcnn_model);
%d.feat = rcnn_scale_features(d.feat, opts.feat_norm_mean);
d.feat = get_feature(rcnn_model, imdb.name, ...
    struct('image_id', imdb.image_ids{ind}, 'IX', []), eigen);
d.feat = rcnn_scale_features(d.feat, opts.feat_norm_mean, rcnn_model);

if isempty(d.feat)
  X_neg = cell(max(class_ids), 1);
  keys = cell(max(class_ids), 1);
  return;
end

fprintf('%d features in image %d\n', size(d.feat, 1), ind);

neg_ovr_thresh = 0.3;

if first_time
  for cls_id = class_ids
    I = find(d.overlap(:, cls_id) < neg_ovr_thresh);
    X_neg{cls_id} = d.feat(I,:);
    keys{cls_id} = [ind*ones(length(I),1) I];
    %fprintf('class %d, neg sample num: %d\n', cls_id, length(I));
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
    %fprintf('class %d: neg sample num: %d\n', cls_id, length(I));
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
function [X_pos, keys, im_IX] = get_positive_pool5_features(rcnn_model, imdb, opts, eigen)
% ------------------------------------------------------------------------
X_pos = cell(max(imdb.class_ids), 1);
keys = cell(max(imdb.class_ids), 1);
im_IX = cell(max(imdb.class_ids), 1);

for i = 1:length(imdb.image_ids)
  tic_toc_print('%s: pos features %d/%d\n', ...
                procid(), i, length(imdb.image_ids));

  d = rcnn_load_cached_pool5_features(opts.cache_name, ...
      imdb.name, imdb.image_ids{i}, false, {'class'});
  key = struct('image_id', imdb.image_ids{i}, 'IX', []);
  feat = get_feature(rcnn_model, imdb.name, key, eigen);

  for j = imdb.class_ids
    if isempty(keys{j})
      X_pos{j} = single([]);
      keys{j} = [];
      im_IX{j} = [];
    end
    sel = find(d.class == j);
    if ~isempty(sel)
      X_pos{j} = cat(1, X_pos{j}, feat(sel,:));
      keys{j} = cat(1, keys{j}, [i*ones(length(sel),1) sel]);
      im_IX{j} = cat(1, im_IX{j}, ... 
          struct('image_id', imdb.image_ids{i}, 'IX', sel)); 
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

function print_array(name, arr)
fprintf('%s = [', name);
for i = 1:length(arr)
  fprintf('%f, ', arr(i));
end
fprintf(']\n');

