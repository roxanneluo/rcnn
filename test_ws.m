function res = test_ws(rcnn_model5, rcnn_model7, imdb, weight,  suffix)
% res = rcnn_test(rcnn_model, imdb, suffix)
%   Compute test results using the trained rcnn_model on the
%   image database specified by imdb. Results are saved
%   with an optional suffix.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

conf = rcnn_config('sub_dir', imdb.name);
image_ids = imdb.image_ids;

% assume they are all the same
feat_opts5 = rcnn_model5.training_opts;
feat_opts7 = rcnn_model7.training_opts;
num_classes = length(rcnn_model5.classes);
fprintf('cache_name7 = %s\n', feat_opts7.cache_name);
fprintf('cache_name5 = %s\n', feat_opts5.cache_name);

if ~exist('suffix', 'var') || isempty(suffix)
  suffix = '';
else
  suffix = ['_' suffix];
end

try
  fprintf('======================try load boxes======================\n');
  aboxes = cell(num_classes, 1);
  for i = 1:num_classes
    fprintf([conf.cache_dir rcnn_model5.classes{i} '_boxes_' imdb.name suffix]);
    load([conf.cache_dir rcnn_model5.classes{i} '_boxes_' imdb.name suffix]);
    aboxes{i} = boxes;
  end
catch
  fprintf('======================catch load boxes======================\n');
  aboxes = cell(num_classes, 1);
  box_inds = cell(num_classes, 1);
  for i = 1:num_classes
    aboxes{i} = cell(length(image_ids), 1);
    box_inds{i} = cell(length(image_ids), 1);
  end

  % heuristic that yields at most 100k pre-NMS boxes
  % per 2500 images
  max_per_set = ceil(100000/2500)*length(image_ids);
  max_per_image = 100;
  top_scores = cell(num_classes, 1);
  thresh = -inf(num_classes, 1);
  box_counts = zeros(num_classes, 1);

  if ~isfield(rcnn_model5, 'folds')
    folds{1} = 1:length(image_ids);
  else
    folds = rcnn_model5.folds;
  end

  count = 0;
  s = 1+weight;
  w7 = 1/s; w5 = weight/s;
  for f = 1:length(folds)
    for i = folds{f}
      count = count + 1;
      fprintf('%s: test (%s) %d/%d\n', procid(), imdb.name, count, length(image_ids));
      d = rcnn_load_cached_pool5_features(feat_opts7.cache_name, ...
          imdb.name, image_ids{i}, true, {'boxes', 'gt'});
      %d.feat = rcnn_pool5_to_fcX(d.feat, feat_opts.layer, rcnn_model);
      %d.feat = rcnn_scale_features(d.feat, feat_opts.feat_norm_mean);
      fc7 = get_feature(d.feat, rcnn_model7, imdb.name, ...
          struct('image_id', image_ids{i}, 'IX', []), []);
      fc7 = rcnn_scale_features(fc7, feat_opts7.feat_norm_mean, rcnn_model7);
      zs7 = bsxfun(@plus, fc7*rcnn_model7.detectors(f).W, rcnn_model7.detectors(f).B);

      conv5 = get_feature(d.feat, rcnn_model5, imdb.name, ...
          struct('image_id', image_ids{i}, 'IX', []), []);
      conv5 = rcnn_scale_features(conv5, feat_opts5.feat_norm_mean, rcnn_model5);
      zs5 = bsxfun(@plus, conv5*rcnn_model5.detectors(f).W, rcnn_model5.detectors(f).B);
      
      zs = w7*zs7+w5*zs5;

      if isempty(d.feat)
        continue;
      end

      for j = 1:num_classes
        boxes = d.boxes;
        z = zs(:,j);
        I = find(~d.gt & z > thresh(j));
        boxes = boxes(I,:);
        scores = z(I);
        aboxes{j}{i} = cat(2, single(boxes), single(scores));
        [~, ord] = sort(scores, 'descend');
        ord = ord(1:min(length(ord), max_per_image));
        aboxes{j}{i} = aboxes{j}{i}(ord, :);
        box_inds{j}{i} = I(ord);

        box_counts(j) = box_counts(j) + length(ord);
        top_scores{j} = cat(1, top_scores{j}, scores(ord));
        top_scores{j} = sort(top_scores{j}, 'descend');
        if box_counts(j) > max_per_set
          top_scores{j}(max_per_set+1:end) = [];
          thresh(j) = top_scores{j}(end);
        end
      end
    end
  end

  for i = 1:num_classes
    % go back through and prune out detections below the found threshold
    for j = 1:length(image_ids)
      if ~isempty(aboxes{i}{j})
        I = find(aboxes{i}{j}(:,end) < thresh(i));
        aboxes{i}{j}(I,:) = [];
        box_inds{i}{j}(I,:) = [];
      end
    end

    save_file = [conf.cache_dir rcnn_model5.classes{i} '_boxes_' imdb.name suffix];
    boxes = aboxes{i};
    inds = box_inds{i};
    save(save_file, 'boxes', 'inds');
    clear boxes inds;
  end
end
fprintf('======================out of catch======================\n');
% ------------------------------------------------------------------------
% Peform AP evaluation
% ------------------------------------------------------------------------
for model_ind = 1:num_classes
  cls = rcnn_model5.classes{model_ind};
  res(model_ind) = imdb.eval_func(cls, aboxes{model_ind}, imdb, suffix);
end

fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Results:\n');
aps = [res(:).ap]';
disp(aps);
disp(mean(aps));
fprintf('~~~~~~~~~~~~~~~~~~~~\n');
