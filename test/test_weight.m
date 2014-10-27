function test_weight(model, layer)
feat_opts = struct('layer', {layer}, 'd', {true}, 'w', {true}, 'combine', @l2, 'combine_name', 'l2');
imdb_name = 'voc_2007_trainval';

model.feat_opts = feat_opts;
model.dims = get_feat_dims(feat_opts);
model.feat_dim = sum(model.dims);
model.exist_r = exist_response(feat_opts);
model.exist_w = exist_weight(feat_opts);
model.equal_dim = false;
model.norm_weight = true;

keys = struct('image_id', '000026', 'IX', []);
pool5 = single(rand(1,9216));
pool5=[];
tic
weight = get_pos_feature(pool5, model, imdb_name, keys);
toc
%{
fc7 = rcnn_pool5_to_fcX(pool5, 7-5, model);
size(fc7)
size(weight)
err = 1e-6;
diff = reshape(abs(fc7-weight(:, 1:4096)), [1, numel(fc7)]);
fprintf('diff max=%f\n', max(diff));
fprintf('diff mean=%f\n', mean(diff));
fprintf('fc7 mean = %f\n', mean(mean(fc7)));
%}
fprintf('weight mean = %f\n', mean(reshape(weight, [numel(weight),1])));
%fprintf('different number=%d\n', sum(diff > err));
weight(:, 20:40)
sum(isnan(weight))
