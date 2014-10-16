function test_model(i, model, num, layer)
%  model = rcnn_create_model('model-defs/pascal_top_entropy.prototxt', './data/caffe_nets/finetune_voc_2007_trainval_iter_70k', 'hhaa', true)
 % model = rcnn_load_model(model, true);
  cell_feat_opts = squeeze(create_feat_opts(i));
  %for i=1:length(cell_feat_opts)
  for i=1:1
    global my_test_feat_opts
    my_test_feat_opts = cell_feat_opts{i};
    sprintf(feat_opts_to_string(my_test_feat_opts))
    conf = rcnn_config('sub_dir', 'haha');
    model.feat_opts = conf.feat_opts;
    pool5 = zeros(num, 9216);
    pool5(:,1) = 1;
    tic;
    feat = rcnn_pool5_to_fcX(pool5, layer-5, model);
    toc;
    tic;
    get_feature_forward(single(pool5), model, layer);
    toc;
    size(feat)
  end
  
%  response = caffe('get_response', 2, true);
 % size(response)
