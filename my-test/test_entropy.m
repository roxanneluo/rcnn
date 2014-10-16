function test_model(i, model, num)
global my_test_feat_opts
my_test_feat_opts = struct('layer', 5, 'd', true, ...
    'w', false, 'combine', @l2, 'combine_name', 'l2'); 
sprintf(feat_opts_to_string(my_test_feat_opts))
conf = rcnn_config('sub_dir', 'haha');
model.feat_opts = conf.feat_opts;
pool5 = single(rand(num, 9216));
feat = get_feature_en(single(pool5), model);
sprintf('feat:')
size(feat)
diff = entropy_diff(pool5, model.cnn.batch_size);
difference = abs(diff-feat);
difference = reshape(difference, [num*9216, 1]);
feat = reshape(feat, [num*9216, 1]);
'pool5 1'
pool5(1,1)
'feat_1'
feat(1)
'diff_1'
diff(1)
'mean_feat'
mean(feat)
'min pool5'
min(min(pool5))
'mean'
mean(difference)
'max'
max(difference)

%  response = caffe('get_response', 2, true);
% size(response)

% combine fun
% input: actually a 3-dim [combine_along_dim, combine_across_dim, num]
% output: a combined vector along dim 2 of size [combine_across_dim, num]
% -----------------------------------------------------------------------------
function feat = l2(diff)
% -----------------------------------------------------------------------------
feat = sqrt(sum(diff.*diff, 1));
feat = normalize(feat);


% -----------------------------------------------------------------------------
% input: a cube [1, combine_across_dim, num]
% output: a matrix without 1
function feat = normalize(feat)
% -----------------------------------------------------------------------------
[along, across, num] = size(feat);
assert(along == 1);
feat = reshape(feat, [across, num]);
err = 1e-30;
s = sqrt(sum(feat.*feat,1));
s(abs(s) < err) = 1;
feat = feat./(ones(size(feat,1), 1)*s);

% -----------------------------------------------------------------------------
function diff = entropy_diff(prob, batch_size)
% -----------------------------------------------------------------------------
num = size(prob, 1);
rest = mod(num, batch_size)
full_num = num-rest;
diff = single(1+log(prob));
diff(1:full_num, :) = diff(1:full_num, :) /batch_size;
if rest ~= 0
  diff(full_num+1:num,:) = diff(full_num+1:num,:)/ rest;
end
diff = -diff;
s = sqrt(sum(diff.*diff,2));
err = 1e-20;
s(abs(s) < err) = 1;
diff = diff./(s*ones(1, size(prob,2)));
diff = abs(diff);


