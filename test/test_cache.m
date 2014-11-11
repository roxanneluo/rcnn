function test_cache(feat_name, dataset)
VOCdevkit = './datasets/VOCdevkit2007';
imdb = imdb_from_voc(VOCdevkit, dataset, '2007');
filename = sprintf('test/%s_err_%s.txt', feat_name, dataset); 
f = fopen(filename, 'w');
for i = 1:length(imdb.image_ids)
  try
    fprintf('Loading from %s %d/%d\n', imdb.image_ids{i}, ...
        i, length(imdb.image_ids));
    d = load_cached_features(feat_name, imdb.name, ...
        imdb.image_ids{i}, false, {'class'});
  catch err
    fprintf('!!!!!!!!!!!!!!!!Error!!!!!!!!!!!!!\n');
    disp(err);
    fprintf('%s\n', err.identifier);
    fprintf('%s\n', err.message);
    fprintf(f, '%s\n', imdb.image_ids{i});
  end
end
fclose(f);
