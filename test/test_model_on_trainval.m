function train_res = test_model_on_trainval(feat_name)
VOCdevkit = './datasets/VOCdevkit2007';
imdb = imdb_from_voc(VOCdevkit, 'trainval', '2007');

model_file = ['cachedir/' feat_name '/voc_2007_trainval/rcnn_model.mat'];
model = rcnn_load_model(model_file);

diary_dir =  ['cachedir/' feat_name '/test_model_on_tranival/'];
system(['mkdir -p ' diary_dir])
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file =[diary_dir 'test_model_on_trainval'...
  timestamp '.txt']
diart(diary_file);
model = add_model_fields(model);
train_res = rcnn_test(model, imdb, '_on_trainval');
