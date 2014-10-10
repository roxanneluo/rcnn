function feature = get_feature(pool5, rcnn_model)
% pool5 is a row vector
% feat = rcnn_pool5_to_fcX(pool5, layer, rcnn_model)
dim = 2;
feat_opts = rcnn_model.feat_opts;
num_feat = length(feat_opts);
feature = [];
for i=1:num_feat
  feature = cat(dim, feature, get_feature_part(pool5, rcnn_model, feat_opts(i)));
end

% -----------------------------------------------------------------------------
% return a single part of the whole feature as specified by feat_opt
function feat = get_feature_part(pool5, rcnn_model, feat_opt)
% -----------------------------------------------------------------------------
% note: cpp layer starts from 0 while matlab starts from 1
layer = feat_opt.layer;
if strcmp(feat_opt.b_or_d, 'blob') % ignore w_or_r
  feat = rcnn_pool5_to_fcX(pool5, layer-5, rcnn_model);
else
  layer = get_layer_id(layer);
  assert(strcmp(feat_opt.b_or_d, 'diff'));

  is_diff = true;
  pool5 = reshape(pool5, [length(pool5),1,1,1]);
  caffe('forward_backward', {pool5});
  if strcmp(feat_opt.w_or_r, 'weight')
    diff = caffe('get_weight', layer, is_diff);
  else 
    assert(strcmp(feat_opt.w_or_r, 'response'));
    diff = caffe('get_response', layer, is_diff);
  end
  diff.layer_name
  diff(1:10)
  diff = diff.blobs{1};
  feat = feat_opt.combine(diff);
end

function layer = get_layer_id(layer)
assert(layer >= 5 && layer <= 10);
switch layer
  case 5
    layer = -1;
  case 6
    layer = 0;
  case 7
    layer = 3;
  otherwise
    layer = layer - 2;
end
