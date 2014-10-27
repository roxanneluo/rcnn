function feat = caffe_weight_features(im, boxes, rcnn_model)
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
assert(batch_size == 1);

feat_opt = rcnn_model.feat_opts;
assert(feat_opt.w && feat_opt.d);
assert(1 <= feat_opt.layer && feat_opt.layer <= 5);
layer = get_layer(feat_opt);

% compute features for each batch of region images
feat = [];
curr = 1;
padding = single(0);
[ww, wh, wc, wn] = size(rcnn_model.cnn.layers(feat_opt.layer).weights{1});
across_dim = wn;
along_dim = ww*wh*wc;
feat = zeros(size(boxes, 1), across_dim, 'single');
fprintf('[%d, %d]\n', size(boxes, 1), across_dim);
for j = 1:length(batches)
  assert(size(batches{j},4)==1);
  caffe('forward_backward', {batches{j}});
  %caffe('forward_backward', {batches{j}; padding});
  res = caffe('get_weight', layer, feat_opt.d);
  blob = res.blobs{1};
  blob = reshape(blob, [along_dim, across_dim, 1]);

  f = combine(blob, feat_opt.combine, false); % I don't normalize when caching TODO

  feat(curr:curr+size(f,1)-1,:) = f;
  curr = curr + batch_size;
end
fprintf('layer_name: %s, size[%d, %d]\n', res.layer_name, ...
    size(boxes, 1), across_dim);

% ---------------------------------------------------------
function layer = get_layer(feat_opt)
% ---------------------------------------------------------
if feat_opt.layer <= 3
  layer = (feat_opt.layer-1)*4;
else
  layer = 8 + 2*(feat_opt.layer-3);
end
