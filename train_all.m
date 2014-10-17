function [res_test, res_train] = train_all(proc, layer)
cell_feat_opts = create_feat_opts(proc, layer);
for i=1:length(cell_feat_opts)
  global my_test_feat_opts
  my_test_feat_opts = cell_feat_opts{i};
  sprintf('~~~~~~~~~~~~~~~~~~~~~~~%s~~~~~~~~~~~~~~~~~~~~~~~', feat_opts_to_string(my_test_feat_opts))
  [res_test, res_train] = rcnn_exp_train_and_test();
  sprintf('[RESULT] test\n')
  'recall'
  res_test.recall
  'prec'
  res_test.prec
  'ap'
  res_test.ap
  'ap_auc'
  res_test.ap_auc
  sprintf('[RESULT] train\n')
  res_train
  caffe('reset');
end
