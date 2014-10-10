function train_all()
cell_feat_opts = create_feat_opts_test();
for i=1:length(cell_feat_opts)
  global my_test_feat_opts
  my_test_feat_opts = cell_feat_opts{i};
  sprintf('~~~~~~~~~~~~~~~~~~~~~~~%s~~~~~~~~~~~~~~~~~~~~~~~', feat_opts_to_string(my_test_feat_opts))
  [res_test, res_train] = rcnn_exp_train_and_test();
  sprintf('[RESULT] test\n')
  res_test
  sprintf('[RESULT] train\n')
  res_train
end
