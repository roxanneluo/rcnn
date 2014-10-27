function draw_detector(feat_name)
%feat_name = feat_opts_to_string(feat_opts);
model_file = ['cachedir/' feat_name '/voc_2007_trainval/rcnn_model.mat'];
model = rcnn_load_model(model_file);

folder1 = 'detector-res/';
folder = [folder1, feat_name, '/'];
system(['mkdir -p ' folder1]);
system(['mkdir -p ' folder]);

W = model.detectors.W;
num_detect = size(W,2);
feat_dim = size(W,1);
x = 1:feat_dim;
for i=1:num_detect
  plot(x, W(:, i));
  title([feat_name ' ' int2str(i)]);
  saveas(gcf, [folder int2str(i) '.jpg']);
end
