function feat = rcnn_features(im, boxes, rcnn_model)
% feat = rcnn_features(im, boxes, rcnn_model)
%   Compute CNN features on a set of boxes.
%
%   im is an image in RGB order as returned by imread
%   boxes are in [x1 y1 x2 y2] format with one box per row
%   rcnn_model specifies the CNN Caffe net file to use.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

% make sure that caffe has been initialized for this model
if rcnn_model.cnn.init_key ~= caffe('get_init_key')
  error('You probably need to call rcnn_load_model');
end
% Each batch contains 256 (default) image regions.
% Processing more than this many at once takes too much memory
% for a typical high-end GPU.
[batches, batch_padding] = rcnn_extract_regions(im, boxes, rcnn_model);
batch_size = rcnn_model.cnn.batch_size;

feat_opt = rcnn_model.feat_opts;
assert(1 <= feat_opt.layer && feat_opt.layer <= 5);
layer = get_layer(feat_opt);

% compute features for each batch of region images
feat_dim = -1;
feat = [];
curr = 1;
padding = single(0);
if feat_opt.w
  [ww, wh, wc, wn] = size(rcnn_model.cnn.layers{layer}.weight{1});
  feat_dim = ww*wh*wc;
end
for j = 1:length(batches)
  if j == length(batches)
    padding = batch_padding;
  end
  valid_num = batch_size - padding;
  % forward propagate batch of region images 
  caffe_func = 'forward';
  if feat_opt.d
    caffe_func = [caffe_func, '_backward'];
  end
  caffe(caffe_func, {batches{j}; padding});
  if feat_opt.w
    res = caffe('get_weight', layer, feat_opt.d);
    blob = res.blobs{1};
    blob = reshape(blob, [feat_dim, wn, 1]);
  else 
    res = caffe('get_response', layer, feat_opt.d);
    blob = res.blobs{1}(:,:,:,valid_num);
    [w, h, c, n] = size(blob);
    feat_dim = w*h;
    blob = reshape(blob, [feat_dim, c, n]);
  end
  f = combine(blob, feat_opt.combine, false); % I don't normalize when caching TODO
  % first batch, init feat_dim and feat
  if j == 1
    fprintf('layer_name: %s, size[%d, %d]\n', res.layer_name, ...
        size(boxes, 1), feat_dim);
    feat = zeros(size(boxes, 1), feat_dim, 'single');
  end

  feat(curr:curr+size(f,2)-1,:) = f';
  curr = curr + batch_size;
end

% ---------------------------------------------------------
function layer = get_layer(feat_opt)
% ---------------------------------------------------------
if feat_opt.layer <= 3
  layer = (feat_opt.layer-1)*4;
