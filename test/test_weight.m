function test_weight(model, layer)
feat_opts = struct('layer', layer, 'd', true, 'w', true, 'combine', @l2, 'combine_name', 'l2');
imdb_name = 'voc_2007_trainval';

model.feat_opts = feat_opts;
model.dims = get_feat_dims(feat_opts);
model.feat_dim = sum(model.dims);
model.exist_r = exist_response(feat_opts);
model.exist_w = exist_weight(feat_opts);

keys = cell(1);
keys{1} = struct('image_id', '000024', 'IX', [1:2:2001]);
tic
weight = get_feature_weight([], model, imdb_name, keys);
toc
size(weight)
