function [train_res, test_res] = test_ws_main(part)
weights = 3.^[-5:1];
if part == 1
  weights = weights(1:3);
else
  weights = weights(4:end);
end

fprintf('!!!!!!!!!!!!!!!!!!testing these weights!!!!!!!!!!!!!!!!!!!!!\n');
disp(weights);

feat_names = {'sb_norm_w_equal_dim_l5_d_w_l2','norm_to_20_d_mean_norm_b_each_part_l7_b_r_l2'};
models = cell(2,1);
for i =1:2
  caffe('reset');
  model_file = ['cachedir/' feat_names{i} '/voc_2007_trainval/rcnn_model.mat'];
  models{i} = rcnn_load_model(model_file);
  models{i} = add_model_fields(models{i});
end
model5 = models{1}
model7 = models{2}

global my_test_feat_opts
my_test_feat_opts = create_feat_opts(6,5);
my_test_feat_opts = my_test_feat_opts{1};

train_res = cell(length(weights), 1);
test_res = cell(length(weights),1);
for i = 1:length(weights)
  VOCdevkit = './datasets/VOCdevkit2007';
  imdb_train = imdb_from_voc(VOCdevkit, 'trainval', '2007');
  imdb_test = imdb_from_voc(VOCdevkit, 'test', '2007');

  global dir_suffix
  weight = weights(i);
  dir_suffix = sprintf('w%f',weight); 
  conf = rcnn_config('sub_dir', imdb_test.name);
  timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
  diary_file = [conf.cache_dir 'ws_test_' timestamp '.txt']
  diary(diary_file);
%{
  fprintf('=====================testing weighted sum, w=%f======================\n', weight);  
  fprintf('==================testing on test =============\n');
  res_test = test_ws(model5, model7, imdb_test,weight,dir_suffix);
  test_res_file = sprintf('%stest_result_w%f', conf.cache_dir, weights(i));
  save(test_res_file, 'res_test');

  fprintf('==================testing on trainval =============\n');
  res_train = test_ws(model5, model7, imdb_train,weight,dir_suffix);
  train_res_file = sprintf('%strain_result_w%f', conf.cache_dir, weights(i));
  save(train_res_file, 'res_train');

  train_res{i} = res_train;
  test_res{i} = res_test;
end

dir = 'cachedir/weighted_sum/';
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = [dir 'results_' timestamp '.txt'];
mkdirs({dir});
diary(diary_file);
for i = 1:length(weights)
  fprintf('\n~~~~~~~~~~weight=%f~~~~~~~~~~\n',weights(i));
  disp('  TRAIN   TEST');
  aps = [[train_res{i}(:).ap]', [test_res{i}(:).ap]'];
  disp(aps);
  disp(mean(aps));
  fprintf('~~~~~~~~~~~~weight=%f done~~~~~~~~\n', weights(i));
  %}
end
