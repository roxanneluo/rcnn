function res_train = exp_model_on_train(feat_name)
net_file     = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';
cache_name   = 'v1_finetune_voc_2007_trainval_iter_70k';
crop_mode    = 'warp';
crop_padding = 16;
layer        = 7;
k_folds      = 0;

% change to point to your VOCdevkit install
VOCdevkit = './datasets/VOCdevkit2007';
imdb_train = imdb_from_voc(VOCdevkit, 'trainval', '2007');

model_file = ['cachedir/' feat_name '/voc_2007_trainval/rcnn_model.mat'];
rcnn_model = rcnn_load_model(model_file);
feat_opts = rcnn_model.feat_opts;
rcnn_model = set_field(rcnn_model, 'dims',      get_feat_dims(feat_opts));
rcnn_model = set_field(rcnn_model, 'feat_dim',  sum(rcnn_model.dims));
rcnn_model = set_field(rcnn_model, 'exist_r',   exist_response(feat_opts));
rcnn_model = set_field(rcnn_model, 'exist_w',   exist_weight(feat_opts));
rcnn_model = set_field(rcnn_model, 'norm_weight', true);
rcnn_model = set_field(rcnn_model, 'equal_dim', true);
rcnn_model

global my_test_feat_opts
my_test_feat_opts = feat_opts

timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = ['cachedir/' feat_name '/test_on_trainval/rcnn_test_on_trainval_' timestamp '.txt'];
system(['mkdir -p cachedir/' feat_name '/test_on_trainval']);
diary(diary_file);
res_train = rcnn_test(rcnn_model, imdb_train, '_trainval');

%------------------------------------------------------------------------------
function model = set_field(model, field, value)
%------------------------------------------------------------------------------
if ~isfield(model, field)
  eval(['model.' field ' = value;']);
end
str = ['model.' field '=%d\n'];
fprintf(str, eval(['model.' field]));
