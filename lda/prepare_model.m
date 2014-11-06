function rcnn_model = prepare_model(feat_opt, opts)
assert(numel(feat_opt) == 1);
assert(~(~feat_opt.d && feat_opt.w));
if feat_opt.d && feat_opt.w
  batch_size = 1;
else 
  batch_size = 128;
end
opts.net_file = './data/nizf/nizf_model';
opts.net_def_file = ['./model-defs/nizf_batch_1_' opts.backward_type '.prototxt'];
rcnn_model = rcnn_create_model(opts.net_def_file, opts.net_file);
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = 'warp';
rcnn_model.detectors.crop_padding = 16;
rcnn_model.feat_opts = feat_opt;
rcnn_model.cnn.batch_size = batch_size;
