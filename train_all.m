function [res_test, res_train] = train_all(proc, layer, norm_weight, equal_dim, proj, whiten, pca_ratio)
if ~exist('proj', 'var')
  proj = false;
  whiten = false;
end
cell_feat_opts = create_feat_opts(proc, layer);

for i=1:length(cell_feat_opts)
  global pca_ratio
  pca_ratio = pca_ratio;
  global proj whiten
  proj = proj;
  whiten = whiten;
  global equal_dim
  equal_dim = equal_dim;
  global norm_weight
  norm_weight = norm_weight;
  global my_test_feat_opts
  my_test_feat_opts = cell_feat_opts{i};
  sprintf('~~~~~~~~~~~~~~~~~~~~~~~%s~~~~~~~~~~~~~~~~~~~~~~~', feat_opts_to_string(my_test_feat_opts))
  [res_test, res_train] = rcnn_exp_train_and_test(norm_weight, equal_dim, proj, whiten, pca_ratio);
  %{
  sprintf('[RESULT] test\n')
  'recall'
  res_test.recall
  'prec'
  res_test.prec
  'ap'
  res_test.ap
  'ap_auc'
  res_test.ap_auc
  %}
  sprintf('[RESULT] train\n')
  res_train
  caffe('reset');
end
