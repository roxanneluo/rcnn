function res_test = exp_test()
% Runs an experiment that trains an R-CNN model and tests it.

% -------------------- CONFIG --------------------
net_file     = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';
cache_name   = 'v1_finetune_voc_2007_trainval_iter_70k';
crop_mode    = 'warp';
crop_padding = 16;
layer        = 7;
k_folds      = 0;

% change to point to your VOCdevkit install
VOCdevkit = './datasets/VOCdevkit2007';
% ------------------------------------------------

imdb_test = imdb_from_voc(VOCdevkit, 'test', '2007');

global my_test_feat_opts
my_test_feat_opts = struct('layer', {7,5}, 'd', {false, true},...
    'w', {false, true}, 'combine', @l2, 'combine_name', 'l2'); 
feat_name = feat_opts_to_string(my_test_feat_opts);

global equal_dim
equal_dim = false;

global norm_weight
norm_weight = true;

feat_name = ['max_' feat_name];
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = ['cachedir/' feat_name '/voc_2007_test/rcnn_test_max_' timestamp '.txt'];
system(['mkdir -p cachedir/' feat_name]);
system(['mkdir -p cachedir/' feat_name '/voc_2007_test']);
diary(diary_file);

feat7_name = 'norm_to_20_d_mean_norm_b_each_part_l7_b_r_l2';
feat5_name = 'norm_to_20_d_mean_norm_b_each_part_l5_d_w_l2';
dir_tail = '/voc_2007_trainval/rcnn_model.mat';
cachedir = 'cachedir/';
rcnn_model5 = rcnn_load_model([cachedir feat5_name dir_tail]);
caffe('reset');
rcnn_model7 = rcnn_load_model([cachedir feat7_name dir_tail]);

rcnn_model7 = upgrade_model(rcnn_model7, imdb_test.classes);
rcnn_model5 = upgrade_model(rcnn_model5, imdb_test.classes);

res_test = rcnn_test_max(rcnn_model5, rcnn_model7, imdb_test, '_max');


%------------------------------------------------------------------------------
function rcnn_model = upgrade_model(rcnn_model, classes)
%------------------------------------------------------------------------------
feat_opts = rcnn_model.feat_opts;
rcnn_model = set_field(rcnn_model, 'dims',      get_feat_dims(feat_opts));
rcnn_model = set_field(rcnn_model, 'feat_dim',  sum(rcnn_model.dims));
rcnn_model = set_field(rcnn_model, 'exist_r',   exist_response(feat_opts));
rcnn_model = set_field(rcnn_model, 'exist_w',   exist_weight(feat_opts));
rcnn_model = set_field(rcnn_model, 'norm_weight', true);
rcnn_model = set_field(rcnn_model, 'equal_dim', false);
%rcnn_model = set_field(rcnn_model, 'classes',   classes);
rcnn_model

%------------------------------------------------------------------------------
function model = set_field(model, field, value)
%------------------------------------------------------------------------------
if ~isfield(model, field)
  eval(['model.' field ' = value;']);
end
str = ['model.' field '=%d\n'];
fprintf(str, eval(['model.' field]));
