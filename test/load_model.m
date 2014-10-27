function model = load_model
model = rcnn_create_model('model-defs/pascal_top_entropy_easy.prototxt', './data/caffe_nets/finetune_voc_2007_trainval_iter_70k', 'hhaa', true)
%model = rcnn_create_model('model-defs/softmax_back.prototxt', './data/caffe_nets/finetune_voc_2007_trainval_iter_70k', 'hhaa', true)
  model = rcnn_load_model(model, true);
%  response = caffe('get_response', 2, true);
 % size(response)
