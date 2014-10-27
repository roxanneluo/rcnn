function feature = get_feature(pool5, rcnn_model, imdb_name, keys)
norm_weight = rcnn_model.norm_weight;
feat_opts = rcnn_model.feat_opts;
num_feat = length(feat_opts);
dims = rcnn_model.dims;
feat_dim = rcnn_model.feat_dim;
exist_r = rcnn_model.exist_r;
exist_w = rcnn_model.exist_w;

if ~isempty(pool5)
  total_num = size(pool5,1);
else
  total_num = 0;
  for i = 1:length(keys)
    total_num = total_num + length(keys(i).IX);
  end
end
% so strange that when IX=[], it still works TODO
feature = single(zeros(total_num, feat_dim));

if exist_r
  pool5 = pool5';
  [pool5_dim, ~] = size(pool5);
  pool5 = reshape(pool5, [1, 1, pool5_dim, total_num]);

  batch_size = rcnn_model.cnn.batch_size;
  [num_batches, batches, num_padding] = get_batches(pool5, batch_size); 

  padding = single(zeros(1,1,1,1));
  for i=1:num_batches
    if i == num_batches
      padding(1,1,1,1) = num_padding;
    end
    num_start = (i-1)*batch_size+1;
    num_end = num_start+batch_size-padding-1;
    
    need_fb = false;
    for j = 1:length(feat_opts)
      need_fb = need_fb || (~feat_opts(j).w && feat_opts(j).d);
    end
    caffe_func = 'forward';
    if need_fb
      caffe_func = [caffe_func,'_backward'];
    end
    caffe(caffe_func, {batches{i}; padding});
    fprintf('%s for batch%d of num%d', caffe_func, i, batch_size-padding);

    dim_start = 1;
    for j = 1:length(feat_opts)
      feat_opt = feat_opts(j);
      if ~feat_opt.w
        feature(num_start:num_end, dim_start:dim_start+dims(j)-1) = ...
          get_feature_part(feat_opts(j), batch_size-padding);
      end
      dim_start = dim_start+dims(j);
    end
  end
end
if exist_w
  num_start = 1;
  for i = 1:length(keys)
    img_id = keys(i).image_id;
    IX = keys(i).IX;

    dim_start = 1;
    for j = 1:num_feat
      dim_end = dim_start+dims(j)-1;
      if feat_opts(j).w
        weight = get_weight_part(imdb_name, img_id, IX, feat_opts(j), norm_weight);
        num_end = num_start+size(weight,1)-1;
        feature(num_start:num_end, dim_start:dim_end) = weight;
        %fprintf('img_id=%s, len(weight)=%d, len(IX) = %d\n', img_id, size(weight,1), length(IX));
      end
      dim_start = dim_end+1;
    end
    num_start = num_end+1;
  end
end

assert(size(size(feature), 2) == 2);
fprintf('size feature [%d, %d]\n', size(feature,1), size(feature,2));


% -----------------------------------------------------------------------------
function weight  = get_weight_part(imdb_name, image_id, IX, feat_opt, norm_weight)
% -----------------------------------------------------------------------------
assert(feat_opt.w && feat_opt.d && feat_opt.layer <= 5);
file = sprintf('./feat_cache/%s/%s/%s.mat', feat_opts_to_string(feat_opt), ...
    imdb_name, image_id);
assert(exist(file, 'file') == 2);
d = load(file, 'feat');
%fprintf('load weight from %s, size=[%d,%d]\n', file, size(d.feat,1), size(d.feat,2));
if ~isempty(IX)
  weight = d.feat(IX,:);
else
  weight = d.feat;
end
if norm_weight
  weight = normalize(weight);
end
%{
if feat_opt.layer > 5
  res = caffe('get_weight', layer, feat_opt.d);
  diff = squeeze(res.blobs{1});
  assert(size(size(diff),2) < 3)
  [w, h] = size(diff);
  diff = reshape(diff, [w,h,1]);
else
%}

% -----------------------------------------------------------------------------
% return a single part of the whole feature as specified by feat_opt
function feat_part = get_feature_part(feat_opt, valid_num)
% -----------------------------------------------------------------------------
% note: cpp layer starts from 0 while matlab starts from 1
%TODO
assert(~feat_opt.w);

%layer = get_layer_id(feat_opt);
layer = -1;
res = caffe('get_response', layer, feat_opt.d);
diff = squeeze(res.blobs{1}(:,:,:,1:valid_num));
[dim, num] = size(diff);
fprintf('layer name: %s, dim:%d, num:%d', res.layer_name, dim, num);
assert(dim > 1);
if feat_opt.layer == 5 && feat_opt.d
  diff = reshape(diff, [36, 256, num]);
else
  diff = reshape(diff, [1, dim, num]);
end
% diff is [combine_along_dim, combine_across_dim, num]
feat_part = combine(diff, feat_opt.combine, false); %TODO

% -----------------------------------------------------------------------------
% input: a matrix [combine_across_dim, num]
% output: a matrix without 1
function feat = normalize(feat)
% -----------------------------------------------------------------------------
err = 1e-20;
s = sqrt(sum(feat.*feat,2));
s(abs(s) < err) = 1;
feat = feat./(s*ones(1, size(feat,2)));
% feat is [num, combine_across_dim]


% -----------------------------------------------------------------------------
function [num_batches, batches, num_padding] = get_batches(pool5, batch_size)
% -----------------------------------------------------------------------------
[w, h, c, n] = size(pool5);
num_batches = ceil(n / batch_size);
num_padding = batch_size-mod(n, batch_size);
if num_padding == batch_size
  num_padding = 0;
end
batches = cell(num_batches, 1);
parfor batch = 1:num_batches
  start = (batch-1)*batch_size+1;
  if batch == num_batches && num_padding ~= 0
    batches{batch} = single(zeros(w, h, c, batch_size));
    batches{batch}(:,:,:,1:batch_size-num_padding) ...
      = pool5(:,:,:,start:n)
  else 
    batches{batch} = pool5(:,:,:, start:start+batch_size-1);
  end
end

% for blob use the layer after relu
% for diff use the layer before relu
function layer = get_layer_id(feat_opt)
layer = feat_opt.layer;
assert(layer >= 5 && layer <= 10);
switch layer
  case 5
    layer = -1;
  case 6
    layer = 1;
  case 7
    layer = 3;
  otherwise
    layer = layer - 4;
end
if feat_opt.d && (feat_opt.layer == 6 || feat_opt.layer == 7)
  layer = layer - 1;
end
