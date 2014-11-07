function cache_features(imdb, feat_opt, varargin)
% rcnn_cache_pool5_features(imdb, varargin)
%   Computes pool5 features and saves them to disk. We compute
%   pool5 features because we can easily compute fc6 and fc7
%   features from them on-the-fly and they tend to compress better
%   than fc6 or fc7 features due to greater sparsity.
%
%   Keys that can be passed in:
%
%   start             Index of the first image in imdb to process
%   end               Index of the last image in imdb to process
%   crop_mode         Crop mode (either 'warp' or 'square')
%   crop_padding      Amount of padding in crop
%   net_file          Path to the Caffe CNN to use
%   cache_name        Path to the precomputed feature cache

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
ip = inputParser;
ip.addRequired('imdb', @isstruct);
ip.addOptional('start', 1, @isscalar);
ip.addOptional('end', 0, @isscalar);
ip.addOptional('crop_mode', 'warp', @isstr);
ip.addOptional('crop_padding', 16, @isscalar);
ip.addOptional('net_file', ...
    'sorry', ...
    @isstr);
ip.addOptional('cache_name', ...
    'oops', @isstr);
ip.addOptional('backward_type', 'no', @isstr);
ip.addOptional('opts',          [],   @isstruct);

ip.parse(imdb, varargin{:});
opts = ip.Results;

assert(numel(feat_opt) == 1);
assert(~(~feat_opt.d && feat_opt.w));
if feat_opt.d && feat_opt.w
  batch_size = 1;
else 
  batch_size = 128;
end
%TODO
%opts.net_def_file = ['./model-defs/pascal_batch' int2str(batch_size) '_output_entropy.prototxt'];
%opts.net_def_file = ['./model-defs/pascal_batch' int2str(batch_size) '_output_softmax_back.prototxt'];
opts.net_def_file = ['./model-defs/nizf_batch_1_' opts.backward_type '.prototxt'];
if opts.opts.do_lda
  trans = load_trans(feat_opts_to_string(feat_opt), opts.opts);  
  proj_dim = cell_size_sum(trans, 2);
  filter_start = get_filter_start(trans);
end

image_ids = imdb.image_ids;
if opts.end == 0
  opts.end = length(image_ids);
end

% Where to save feature cache
opts.output_dir = ['./feat_cache/' opts.cache_name '/' imdb.name '/'];
fprintf('[feat cache dir] %s\n', opts.output_dir);
mkdir_if_missing(opts.output_dir);

% Log feature extraction
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = [opts.output_dir 'rcnn_cache_' feat_opts_to_string(feat_opt) ...
              '_features_' timestamp '.txt'];
diary(diary_file);
fprintf('Logging output in %s\n', diary_file);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Feature caching options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

% load the region of interest database
roidb = imdb.roidb_func(imdb);

rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file, opts.cache_name);
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = opts.crop_mode;
rcnn_model.detectors.crop_padding = opts.crop_padding;
rcnn_model.feat_opts = feat_opt;
rcnn_model.cnn.batch_size = batch_size;

total_time = 0;
count = 0;
neg_per_im = 10;
for i = opts.start:opts.end
  fprintf('%s: cache features: %d/%d\n', procid(), i, opts.end);

  save_file = [opts.output_dir image_ids{i} '.mat'];
  if exist(save_file, 'file') ~= 0
    fprintf(' [already exists]\n');
    continue;
  end
  count = count + 1;

  tot_th = tic;

  d = roidb.rois(i);
  im = imread(imdb.image_at(i));
%{  
  IX = sample_neg(neg_per_im, d);
  d.boxes = d.boxes(IX,:); d.class = d.class(IX); 
  d.overlap = d.overlap(IX, :); d.gt = d.gt(IX);
%}
  th = tic;
  d.feat = weight_features(im, d.boxes, d.class, rcnn_model, opts.backward_type,...
      opts.opts);
  if opts.opts.do_normalize
    fprintf('\tnorm');
    d.feat = normalize(d.feat);
  end
  if opts.opts.do_lda
    fprintf('\tlda');
    d.feat = lda(d.feat, trans, proj_dim, filter_start); 
  end
  fprintf('\n [features: %.3fs]\n', toc(th));

  th = tic;
  save(save_file, '-struct', 'd');
  fprintf(' [saving:   %.3fs] [%d,%d]\n', toc(th), size(d.feat,1), size(d.feat,2));

  total_time = total_time + toc(tot_th);
  fprintf(' [avg time: %.3fs (total: %.3fs)]\n', ...
      total_time/count, total_time);
end

function IX = sample_neg(neg_per_im, d)
num_pos = max(find(d.gt));
num_neg = length(d.boxes) - num_pos;
neg_per_im = min(neg_per_im, num_neg);
IX = [1:num_pos+neg_per_im]';
IX(num_pos+1:end) = num_pos+randperm(num_neg, neg_per_im);

