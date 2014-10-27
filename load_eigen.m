function eigen = load_eigen(rcnn_model)
% ------------------------------------------------------------------------
feat_opts = rcnn_model.feat_opts;
if length(feat_opts) == 1 %TODO
  if feat_opts(1).layer == 5
    eigen_file = 'draw-res/pca/sb_norm_w_equal_dim_l5_d_w_l2/0/eigen_projdim0_numpcls5000.mat' ;
  elseif feat_opts(1).layer == 7
    eigen_file = 'draw-res/pca/norm_to_20_d_mean_norm_b_each_part_l7_b_r_l2/eigen_projdim0_numpcls5000.mat';
  else 
    assert(false);
  end
else
  assert(length(feat_opts) == 2 ...
      && ~feat_opts(1).d && feat_opts(1).layer == 7 && ~feat_opts(1).w ...
      && feat_opts(2).d && feat_opts(2).layer == 5 && feat_opts(2).w);
  eigen_file = 'draw-res/pca/sb_norm_w_equal_dim_l7_b_r_l2+l5_d_w_l2/eigen_projdim0_numpcls5000.mat';
end
assert(exist(eigen_file, 'file') == 2);
eigen = load(eigen_file);
