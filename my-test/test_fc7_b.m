function res_test = test_fc7_b
cell_feat_opts = create_feat_opts(0);
feat_opts = cell_feat_opts{1};
global my_test_feat_opts
my_test_feat_opts = feat_opts;
sprintf('~~~~~~~~~~~~~~~~~~~~~~~%s~~~~~~~~~~~~~~~~~~~~~~~', feat_opts_to_string(my_test_feat_opts))


opts = struct;
% -------------------- CONFIG --------------------
opts.net_file     = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';
opts.cache_name   = 'v1_finetune_voc_2007_trainval_iter_70k';
opts.crop_mode    = 'warp';
opts.crop_padding = 16;
opts.layer        = 7;
opts.k_folds      = 0;

% change to point to your VOCdevkit install
VOCdevkit = './datasets/VOCdevkit2007';
% ------------------------------------------------
imdb_test = imdb_from_voc(VOCdevkit, 'test', '2007');


opts.net_def_file = './model-defs/pascal_top_entropy_easy.prototxt';

conf = rcnn_config('sub_dir', imdb_test.name);

% ------------------------------------------------------------------------
% Create a new rcnn model
rcnn_model_file = './data/rcnn_models/voc_2007/rcnn_model_finetuned.mat';
rcnn_model = rcnn_load_model(rcnn_model_file, conf.use_gpu);
rcnn_model
rcnn_model.cnn
rcnn_model.cnn.definition_file = opts.net_def_file;
rcnn_model.cnn.batch_size = 2048;
rcnn_model = rcnn_load_model(rcnn_model, conf.use_gpu);
rcnn_model
rcnn_model.cnn
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Get the average norm of the features
rcnn_model.feat_opts = conf.feat_opts;
fprintf('average norm = %.3f\n', rcnn_model.training_opts.feat_norm_mean);

res_test = rcnn_test(rcnn_model, imdb_test);
fprintf('[TEST RESULT]:\n');
res_test
