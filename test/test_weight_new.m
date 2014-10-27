function test_weight_new(layer)
feat_opts = struct('layer', 5, 'd', true, 'w', true, ...
    'combine', @l2, 'combine_name', 'l2');
model = struct();
model.feat_opts = feat_opts;
model.dims = get_feat_dims(feat_opts);
model.feat_dim = sum(model.dims);
model.exist_r = exist_response(feat_opts);
model.exist_w = exist_weight(feat_opts);
model.equal_dim = false;
model.norm_weight = true;
model.cnn = struct();
model.cnn.batch_size = 1;
model.cnn.input_size = 227;

data_mean_file = './external/caffe/matlab/caffe/ilsvrc_2012_mean.mat';
assert(exist(data_mean_file, 'file') ~= 0);
ld = load(data_mean_file);
image_mean = ld.image_mean; clear ld;
off = floor((size(image_mean,1) - model.cnn.input_size)/2)+1;
image_mean = image_mean(off:off+model.cnn.input_size-1, off:off+model.cnn.input_size-1, :);
model.cnn.image_mean = image_mean;
model.detectors = struct;
model.detectors.crop_mode = 'warp';
model.detectors.crop_padding = 16;


% change to point to your VOCdevkit install
VOCdevkit = './datasets/VOCdevkit2007';
imdb = imdb_from_voc(VOCdevkit, 'trainval', '2007');

id = 1;
box_id = 1;

[weight, w_boxes] = get_feature_w([], model, imdb.name, struct('image_id', imdb.image_ids{id}, 'IX', 6));
'size weight'
size(weight)

imdb.image_ids = imdb.image_ids{id};
roidb = imdb.roidb_func(imdb);
d = roidb.rois(id);
im = imread(imdb.image_at(id));
box = d.boxes(box_id,:);
[batches, batch_padding] = rcnn_extract_regions(im, box, model);
net_def = ['./model-defs/pascal_batch' int2str(model.cnn.batch_size) '_output_entropy.prototxt'];
net_file = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';
init_key = caffe('init', net_def, net_file);
caffe('set_mode_gpu');
caffe('set_phase_train');
caffe('forward_backward', {batches{1};single(0)});
c_weights = caffe('get_weights_diff');
c_weights(layer).layer_names
c_weight = c_weights(layer).weights{1};
size(c_weight)
across = 256;
along = numel(c_weight)/across;
c_weight = reshape(c_weight, [along, across, 1]);
c_weight = combine(c_weight, feat_opts.combine, true);
size(c_weight)

diff = abs(c_weight-weight);
diff = reshape(diff, [1, numel(diff)]);

err = 1e-6;
fprintf('diff max=%f\n', max(diff));
fprintf('diff mean=%f\n', mean(diff));
fprintf('weight mean = %f\n', mean(reshape(weight, [numel(weight), 1])));
fprintf('c_weight mean = %f\n', mean(reshape(weight, [numel(c_weight),1])));
fprintf('different number=%d\n', sum(diff > err));
weight(1:10)
c_weight(1:10)
[lia, loc] = ismember(box, w_boxes, 'rows')
w_boxes(6,:)
