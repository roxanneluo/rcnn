function feat = weight_features(im, boxes, labels, rcnn_model, backward_type, opts)
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
%dims = get_feat_dim(feat_opt);
%feat_dim = prod(dims);
%feat = zeros(size(boxes, 1), feat_dim, 'single');
%fprintf('[%d, %d]\n', size(boxes, 1), feat_dim);
if strcmp(backward_type, 'sl')
  labels = single(labels);
  input2 = labels;
elseif strcmp(backward_type, 'sb')
  input2 = single(zeros(size(boxes,1),1));
else
  assert(false);
end
for j = 1:length(batches)
  assert(size(batches{j},4)==1);
  caffe('forward_backward', {batches{j}; input2(j)});
  res = caffe('get_weight', layer, feat_opt.d);
  blob = res.blobs{1};

  if j == 1
    feat_dim = numel(blob);
    feat = zeros(size(boxes, 1), feat_dim, 'single');
    fprintf('[%d, %d]\n', size(boxes, 1), feat_dim);
  end

  blob = reshape(blob, [feat_dim, 1]);

  feat(j,:) = blob';
end
if isfield(opts, 'do_lda') && opts.do_lda
  if isfield(opts, 'do_normalize') && opts.do_normalize
    feat = normalize(feat);
  end
  feat = lda(feat, opts.trans);
end
fprintf('layer_name: %s, max=%d, size[%d, %d]\n', res.layer_name, ...
    max(max(feat)), size(feat, 1), size(feat, 2));

% ---------------------------------------------------------
function layer = get_layer(feat_opt)
% ---------------------------------------------------------
if feat_opt.layer <= 3
  assert(feat_opt.w && feat_opt.d);
  layer = (feat_opt.layer-3)*6;
else
  switch feat_opt.layer
  case 4
    assert(feat_opt.d && feat_opt.w);
    layer = 16;
  case 5
    if feat_opt.w
      layer = 22;
    else
      layer = 26;
    end
  case 6
    if feat_opt.w
      layer = 27;
    else 
      layer = 28;
    end
  case 7
    if feat_opt.w
      layer = 29
    else
      layer = 30;
    end
  case 8
    layer = 31;
  case 9
    layer = 32;
  end
  % for conv layer cccpX's layer = conv's layer + 2; however for cccp4x, layer = 20
end
