function model = test_model()
  model = rcnn_create_model('model-defs/pascal_top_entropy.prototxt', './data/caffe_nets/finetune_voc_2007_trainval_iter_70k', 'hhaa', true)
  model = rcnn_load_model(model, false);
  cell_feat_opts = create_feat_opts();
  for i=1:length(cell_feat_opts)
    global my_test_feat_opts
    my_test_feat_opts = cell_feat_opts{i};
    conf = rcnn_config('sub_dir', 'haha');
    model.feat_opts = conf.feat_opts;
    feat = get_feature(single(ones(1,9216)), model);
    feat(1:10)
    size(feat)
  end
  
%  response = caffe('get_response', 2, true);
 % size(response)
