function meam_norm = test_stats(layer)
combine = @l2;
combine_name = 'l2';
global my_test_feat_opts 
my_test_feat_opts = struct('layer', layer, 'd', false, ...
    'w', false, 'combine', combine, 'combine_name', combine_name);

net_file     = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';
cache_name   = 'v1_finetune_voc_2007_trainval_iter_70k';
crop_mode    = 'warp';
crop_padding = 16;
layer        = 7;
k_folds      = 0;

% change to point to your VOCdevkit install
VOCdevkit = './datasets/VOCdevkit2007';
% ------------------------------------------------

imdb_train = imdb_from_voc(VOCdevkit, 'trainval', '2007');

[rcnn_model, rcnn_k_fold_model] = ...
    test_rcnn_train(imdb_train, ...
      'layer',        layer, ...
      'k_folds',      k_folds, ...
      'cache_name',   cache_name, ...
      'net_file',     net_file, ...
      'crop_mode',    crop_mode, ...
      'crop_padding', crop_padding);


% -----------------------------------------------------------------------------
function feat = l2(diff)
% -----------------------------------------------------------------------------
feat = sqrt(sum(diff.*diff, 1));

