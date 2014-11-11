function feature = get_feature(rcnn_model, imdb_name, keys, eigen)
norm_weight = rcnn_model.norm_weight;
feat_opts = rcnn_model.feat_opts;
num_feat = length(feat_opts);
dims = rcnn_model.dims;
feat_dim = rcnn_model.feat_dim;
proj = rcnn_model.proj;

%fprintf('proj=%d\n', proj);
if proj
  dims = eigen.scale.dims;
  feat_dim = eigen.scale.feat_dim;
end

total_num = 0;
for i = 1:length(keys)
  total_num = total_num + length(keys(i).IX);
end
% so strange that when IX=[], it still works TODO
feature = single(zeros(total_num, feat_dim));

num_start = 1;
for i = 1:length(keys)
  img_id = keys(i).image_id;
  IX = keys(i).IX;

  dim_start = 1;
  for j = 1:num_feat
    dim_end = dim_start+dims(j)-1;
    feat_opt = feat_opts(j);
    norm_feat = feat_opt.d && (~feat_opt.w || norm_weight);
    %fprintf('norm_feat%d= %d\n', j, norm_feat);
    feat = get_feat_part(imdb_name, img_id, IX, feat_opt, norm_feat);
  %  fprintf('size feat_part(%d,%d)=[%d,%d]\n', i,j,size(feat,1), size(feat,2));
    num_end = num_start+size(feat,1)-1;
   % fprintf('num_start=%d, num_end=%d, dim_start=%d, dim_end=%d\n', num_start, ...
        %num_end, dim_start, dim_end);
    feature(num_start:num_end, dim_start:dim_end) = feat;
    dim_start = dim_end+1;
  end
  num_start = num_end+1;
end

if proj
  assert(~isempty(eigen));
  feature = rcnn_scale_features(feature, eigen.scale.mean_norm, eigen.scale); 
  feature = project(feature, eigen.eigen_vec, eigen.eigen_val, rcnn_model.whiten);
end

assert(size(size(feature), 2) == 2);
fprintf('size feature [%d, %d]\n', size(feature,1), size(feature,2));


% -----------------------------------------------------------------------------
function feat  = get_feat_part(imdb_name, image_id, IX, feat_opt, norm_feat)
% -----------------------------------------------------------------------------
file = sprintf('./feat_cache/%s/%s/%s.mat', feat_opts_to_string(feat_opt), ...
    imdb_name, image_id);
fprintf(file);
assert(exist(file, 'file') == 2);
d = load(file, 'feat');
if ~isempty(IX)
  feat = d.feat(IX,:);
else
  feat = d.feat;
end
if norm_feat
  fprintf('normalize\n');
  feat = normalize(feat);
end

