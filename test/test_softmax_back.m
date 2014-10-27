function test_softmax_back(model, num)
global my_test_feat_opts
my_test_feat_opts = struct('layer',7, 'd', true, ...
    'w', false, 'combine', @l2, 'combine_name', 'l2'); 
sprintf(feat_opts_to_string(my_test_feat_opts))
conf = rcnn_config('sub_dir', 'haha');
model.feat_opts = conf.feat_opts;
model.dims = get_feat_dims(model.feat_opts);
model.feat_dim = sum(model.dims);
model.exist_r = exist_response(model.feat_opts);
model.exist_w = exist_weight(model.feat_opts);
model.norm_weight = false;
model.cnn.batch_size = 200;

dim = 4096;
pool5 = single(rand(num, dim));
feat = get_feature_sb(pool5, model, '', []);
fprintf('feat:');
size(feat)
prob = softmax(pool5);
size(prob)
size(feat)
%diff = cat(2, prob, prob);
diff = prob;
difference = abs(diff-feat);
difference = reshape(difference, [numel(difference), 1]);
feat = reshape(feat, [numel(feat), 1]);
fprintf('pool5_1 = %f\n', pool5(1,1));
fprintf('feat_1 = %f\n', feat(1,1));
fprintf('diff_1 = %f\n', diff(1));
fprintf('mean_feat = %f\n', mean(feat));
fprintf('min_pool5 = %f\n', min(min(pool5)));
fprintf('mean = %f\n', mean(difference))
fprintf('max = %f\n', max(difference));


function prob = softmax(feat)
feat = feat - repmat(max(feat, [], 2), [1, size(feat, 2)]);
exponent = exp(feat);
s = sum(exponent, 2);
prob = exponent ./ repmat(s, [1, size(feat,2)]);
