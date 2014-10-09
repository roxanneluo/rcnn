function feature = get_feature(pool5, rcnn_model, feat_opts)
% feat = rcnn_pool5_to_fcX(pool5, layer, rcnn_model)
dim = 1;
num_feat = size(feat_opts);
feature = [];
for i=1:num_feat
  feature = cat(dim, feature, get_feature_part(pool5, rcnn_model, feat_opts(i)));
end
end

% -----------------------------------------------------------------------------
% return a single part of the whole feature as specified by feat_opt
function feat = get_feature_part(pool5, rcnn_model, feat_opt)
% -----------------------------------------------------------------------------
if strcmp(feat_opt.b_or_d, 'blob') % ignore w_or_r
  feat = rcnn_pool5_to_fcX(pool5, rcnn_model, feat_opt.layer);
else
  assert(strcmp(feat_opt.b_or_d, 'diff'));
  top_feat = rcnn_pool5_to_fcX(pool5, rcnn_model, rcnn_model.top_layer_id)
  if strcmp(feat_opt.w_or_r, 'weight')
    diff = backward_weight(top_feat, rcnn_model, feat_opt.layer);
  else 
    assert(feat_opt.w_or_r, 'response');
    diff = backward(top_feat, rcnn_model, feat_opt.layer);
  end
  feat = feat_opt.combine(diff);
end
end

% -----------------------------------------------------------------------------
% return the top_diff of layer_id
function diff =  backward(top_feat, rcnn_model, layer_id)
% -----------------------------------------------------------------------------
  prob = softmax(top_feat);
  diff = entropy_diff(prob);
  
  if 
end

% -----------------------------------------------------------------------------
function diff = entropy_diff(prob)
% -----------------------------------------------------------------------------
  N = length(prob); % TODO(xuan): I assume it to be a 1-dim vector so far
  s = sum((1+log(prob)).*prob);
  diff = (prob*s-(1+log(prob)).*prob)/N;
end
