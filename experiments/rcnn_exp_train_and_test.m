function [res_test, res_train, rcnn_model] = rcnn_exp_train_and_test(norm_weight, equal_dim, proj, whiten, pca_ratio, svm_C)
% Runs an experiment that trains an R-CNN model and tests it.

% -------------------- CONFIG --------------------
%net_file     = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';
net_file = './data/nizf/nizf_model';
cache_name   = 'l7_b_r_l2';
%cache_name   = 'v1_finetune_voc_2007_trainval_iter_70k';
crop_mode    = 'warp';
crop_padding = 16;
layer        = 7;
k_folds      = 0;
if ~exist('svm_C', 'var')
  svm_C = 10^(-3);
end
if ~exist('proj', 'var')
  whiten = false;
  proj = false;
end
if ~exist('pca_ratio', 'var')
  pca_ratio = 1;
end
% change to point to your VOCdevkit install
VOCdevkit = './datasets/VOCdevkit2007';
% ------------------------------------------------

imdb_train = imdb_from_voc(VOCdevkit, 'trainval', '2007');
imdb_test = imdb_from_voc(VOCdevkit, 'test', '2007');

[rcnn_model, rcnn_k_fold_model] = ...
    rcnn_train(imdb_train, ...
      'layer',        layer, ...
      'k_folds',      k_folds, ...
      'cache_name',   cache_name, ...
      'net_file',     net_file, ...
      'crop_mode',    crop_mode, ...
      'crop_padding', crop_padding, ...
      'norm_weight',  norm_weight, ...
      'equal_dim',    equal_dim, ...
      'proj',         proj, ...
      'whiten',       whiten, ...
      'pca_ratio',    pca_ratio,...
      'svm_C',        svm_C);

if k_folds > 0
  res_train = rcnn_test(rcnn_k_fold_model, imdb_train);
else
  res_train = [];
end
%TODO
res_test = rcnn_test(rcnn_model, imdb_test);
