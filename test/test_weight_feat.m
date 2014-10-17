function test_fcx(model, num, layer)
%TODO
global my_test_feat_opts
my_test_feat_opts = struct('layer', {5},'d', {true}, ...
    'w', true, 'combine', @l2, 'combine_name', 'l2'); 
sprintf(feat_opts_to_string(my_test_feat_opts))
conf = rcnn_config('sub_dir', 'haha');
model.feat_opts = conf.feat_opts;
model.dims = get_feat_dims(model.feat_opts);
model.feat_dim = sum(model.dims);

pool5 = single(rand(num, 9216));
feat = get_feature(single(pool5), model);
fcx1 = rcnn_pool5_to_fcX(pool5, 7-5, model);
fcx2 = rcnn_pool5_to_fcX(pool5, 5-5, model);
fcx = cat(2, fcx1, fcx2);
diff = abs(feat - fcx);
[num, dim] = size(diff);
diff = reshape(diff, [1, num*dim]);
err = 1e-6;
'equal num'
sum(diff < err)
'max'
max(diff)
'mean diff'
mean(diff)
fprintf('mean feat = %f, mean fcx = %f\n', mean(mean(feat)), mean(mean(fcx)));

'fc'
fcx(1:10)

'feat'
feat(1:10)


whos
