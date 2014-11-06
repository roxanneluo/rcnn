function d = compute_grad(rcnn_model, feat_opt, im_id, imdb, roidb, opts)
opts.do_lda = false; opts.do_normalize = false;
tot_th = tic;

d = roidb.rois(im_id);
im = imread(imdb.image_at(im_id));

IX = sample_neg(opts.neg_per_im, d);
d.boxes = d.boxes(IX,:); d.class = d.class(IX); 
d.overlap = d.overlap(IX, :); d.gt = d.gt(IX);

if ~isempty(IX)
  th = tic;
  d.feat = weight_features(im, d.boxes, d.class, rcnn_model, opts.backward_type,...
      opts);
  fprintf(' [features: %.3fs]\n', toc(th));
else 
  fprintf('IX is empty (no positive example)\n');
end


function IX = sample_neg(neg_per_im, d)
num_pos = max(find(d.gt));
num_neg = length(d.boxes) - num_pos;
neg_per_im = min(neg_per_im, num_neg);
IX = [1:num_pos+neg_per_im]';
if neg_per_im > 0
  IX(num_pos+1:end) = num_pos+randperm(num_neg, neg_per_im);
end
