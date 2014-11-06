function test_detec_norm(feat_name)
model_file = ['cachedir/' feat_name '/voc_2007_trainval/rcnn_model.mat'];
model = load(model_file);
model = model.rcnn_model;
W = model.detectors.W;
norms = sum(W.*W);
disp(feat_name);
disp(norms);
disp(sum(norms));
