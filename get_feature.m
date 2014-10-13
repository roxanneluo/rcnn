function feature = get_feature(pool5, rcnn_model)
feat_opts = rcnn_model.feat_opts;
num_feat = length(feat_opts);
dims = get_feat_dims(feat_opts);
feat_dim = sum(dims);

pool5 = pool5';
[pool5_dim, total_num] = size(pool5)
pool5 = reshape(pool5, [pool5_dim, 1, 1, total_num]);

batch_size = rcnn_model.cnn.batch_size;
[num_batches, batches, num_padding] = get_batches(pool5, batch_size); 

padding = single(zeros(1,1,1,1));
feature = zeros(total_num, feat_dim);
for i=1:num_batches
  if i == num_batches
    padding(1,1,1,1) = num_padding;
  end
  num_start = (i-1)*batch_size+1;
  num_end = num_start+batch_size-padding-1;
  
  need_fb = false;
  for j = 1:length(feat_opts)
    if feat_opts(j).d
      need_fb = true;
      break;
    end
  end
  caffe_func = 'forward';
  if need_fb
    caffe_func = [caffe_func,'_backward'];
  end
  caffe(caffe_func, {batches{i}; padding});
  sprintf('%s for batch%d of num%d', caffe_func, i, batch_size-padding)

  dim_start = 1;
  for j = 1:length(feat_opts)
    feat_opt = feat_opts(j);
    feature(num_start:num_end, dim_start:dim_start+dims(j)-1) = ...
      get_feature_part(feat_opts(j), batch_size-padding);
    dim_start = dim_start+dims(j);
  end
end

'size_feature'
size(feature)

% -----------------------------------------------------------------------------
function dims = get_feat_dims(feat_opts)
% -----------------------------------------------------------------------------
dims = zeros(length(feat_opts),1);
for i = 1:length(feat_opts)
  dims(i) = get_feat_part_dim(feat_opts(i));
end

function part_dim = get_feat_part_dim(feat_opt)
  switch(feat_opt.layer)
  case 5
    if feat_opt.d
      part_dim = 256;
    else 
      part_dim = 9216;
    end
  case 6
    part_dim = 4096;
  case 7
    part_dim = 4096;
  case 8
    part_dim = 21;
  otherwise
    assert(false);
  end

% -----------------------------------------------------------------------------
% return a single part of the whole feature as specified by feat_opt
function feat_part = get_feature_part(feat_opt, valid_num)
% -----------------------------------------------------------------------------
% note: cpp layer starts from 0 while matlab starts from 1
layer = get_layer_id(feat_opt);
if feat_opt.w
  res = caffe('get_weight', layer, feat_opt.d);
  diff = squeeze(res.blobs{1});
  if (size(size(diff),2) < 3)
    [w, h] = size(diff);
    diff = reshape(diff, [w,h,1]);
  end
else 
  res = caffe('get_response', layer, feat_opt.d);
  diff = squeeze(res.blobs{1}(:,:,:,1:valid_num));
  [dim, num] = size(diff);
  if feat_opt.layer == 5 && feat_opt.d
    diff = reshape(diff, [36, 256, num]);
  else
    diff = reshape(diff, [1, dim, num]);
  end
end
% diff is [combine_along_dim, combine_across_dim, num]
feat_part = feat_opt.combine(diff);
size(feat_part)
% feat so far is [combine_across_dim, num]
feat_part = feat_part';
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
diff = feat_opt.d;
assert(layer >= 5 && layer <= 10);
switch layer
  case 5
    layer = -1;
  case 6
    layer = 1;
  case 7
    layer = 4;
  otherwise
    layer = layer - 2;
end
if layer == 6 || layer == 7
  layer = layer - 1;
end
