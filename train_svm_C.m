function [train_res, test_res] = train_svm_C(part_id)
fprintf('test the following SVM_C\n')
svm_Cs = 10^(-3)*2.^[-4:1];
if part_id == 1
  svm_Cs = svm_Cs(1:3)
else 
  svm_Cs = svm_Cs(4:end)
end
num_C = length(svm_Cs)

global equal_dim norm_weight my_test_feat_opts
equal_dim         = true;
norm_weight       = true;
cell_feat_opts = create_feat_opts(6,5);
my_test_feat_opts = cell_feat_opts{1};

system('mkdir -p svm');
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = ['svm/log_' timestamp '.txt'];
diary(diary_file);
for i = 1:num_C
  global svm_C
  svm_C = svm_Cs(i);
  fprintf('===========================testing svm_C = %f===========================\n', svm_Cs(i));
  fprintf('\n~~~~~~~~~~~training~~~~~~~~~\n');
  [test_res{i}, ~, rcnn_model ] = ...
    rcnn_exp_train_and_test(norm_weight, equal_dim, false, false, 1, svm_C);

  VOCdevkit = './datasets/VOCdevkit2007';
  imdb_train = imdb_from_voc(VOCdevkit, 'trainval', '2007');
  fprintf('\n~~~~~~~~testing svm_C=%f on trainval~~~~~~~~~~~~\n', svm_C);
  train_res{i} = rcnn_test(rcnn_model, imdb_train);
  fprintf('===========================testing svm_C = %f done===========================\n', svm_C);
  file_name = sprintf('svm/%s_%f', feat_opts_to_string(my_test_feat_opts), svm_C);
  train_result = train_res{i}; test_result = test_res{i};
  save(file_name, 'svm_C', 'train_result', 'test_result');
  caffe('reset');
end

timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = ['svm/log_result_' timestamp '.txt'];
diary(diary_file);
for i = 1:num_C
  fprintf('\n~~~~~~~svm_C=%f~~~~~~\n', svm_Cs(i));
  fprintf('TRAINING   TESTING\n');
  disp([[train_res{:}.ap]', [test_res{:}.ap]']);
  fprintf('\n'); 
  train_ap = train_res{i}(:).ap;
  test_ap = test_res{i}(:).ap;
  fprintf('%f\t\t%f\n', mean(train_ap), mean(test_ap));
end
