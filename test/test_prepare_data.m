%-------------------------------------------------------
function [data, nums, keys, index] = prepare_data(feat_name, opts, dir)
%-------------------------------------------------------
file_name = sprintf('%sORIDATA-npc%d-nbg%d.mat', ...
    dir, opts.max_num_per_class, opts.max_num_bg); 
try
  fprintf('Loading original data from %s\n', file_name);
  ld = load(file_name);
  data = ld.data; nums = ld.nums; clear ld;
  fprintf('Loaded original data from %s\n', file_name);
catch
  fprintf('Calculate data\n');
  VOCdevkit = './datasets/VOCdevkit2007';
  imdb = imdb_from_voc(VOCdevkit, 'train', '2007');

  class_ids = [imdb.class_ids, length(imdb.class_ids)+1];
  classes = {imdb.classes{:}, 'background'};
  feats = cell(length(class_ids), 1);
  neg_ovr_thresh = 0.3;
  exceed = zeros(length(class_ids),1);
  max_check = load('num_sample_per_class.txt')
  if opts.max_num_per_class > 0
    max_check = min(opts.max_num_per_class, max_check);
  end
  max_check = [max_check; opts.max_num_bg_check];
  len_image = length(imdb.image_ids);
  im_ids = randperm(len_image);
  key_cell = cell(length(class_ids),1);
  index_cell = cell(length(class_ids), 1);
  for k = 1:length(im_ids)
    i = im_ids(k);
    if all(exceed([7,15]))
      fprintf('all classes reaches max_num_per_class = %d, %d\n', ...
          opts.max_num_per_class, opts.max_num_bg_check);
      break;
    end

    tic_toc_print('%s: features from %s, %d/%d\n', ...
                  procid(), imdb.image_ids{i}, k, length(im_ids));
    d = load_cached_features(feat_name, ...
        imdb.name, imdb.image_ids{i}, true, {'class'; 'overlap'});

    for j = [7,15]
      if exceed(j)
        continue;
      end
      if isempty(feats{j})
        feats{j} = single([]);
        key_cell{j} = cell(0);
      end
      if j == length(class_ids)
        sel = find(d.class == 0  & all(d.overlap < neg_ovr_thresh, 2)); 
        len = min(10, length(sel));
        if len < length(sel)
          selIX = randperm(length(sel), len);
          sel = sel(selIX);
        end
      else
        sel = find(d.class == j);
      end
      if ~isempty(sel)
        feats{j} = cat(1, feats{j}, d.feat(sel,:));
        key_cell{j} = cat(1, key_cell{j}, rep(imdb.image_ids{i}, length(sel)));
        index_cell{j} = cat(1, index_cell{j}, sel);
        fprintf('len key_cell{%d} = %d, len index_cell{%d} = %d, len index_cell{%d}=%d\n',...
            j, length(key_cell{j}), j, length(index_cell{j}), j, length(index_cell{j}));
        disp(d.class(sel)');
        if max_check(j) >= 0 && size(feats{j},1) >= max_check(j)
          exceed(j) = true;
          feats{j} = feats{j}(1:max_check(j),:);
          key_cell{j}(max_check(j):end) = [];
          index_cell{j}(max_check(j):end) = [];
          if j == max(class_ids) && opts.max_num_bg_check >=0 && opts.max_num_bg >= 0
            IX = randperm(opts.max_num_bg_check, opts.max_num_bg);
            feats{j} = feats{j}(IX,:);
          end
          fprintf('class %d-%s exceeds max_check=%d\n', i, classes{j}, max_check(j));
        end
      end
    end
    print_num(feats);
  end
  data = [];
  nums = zeros(length(class_ids),1);
  keys = cell(0); index = [];
  for j = class_ids 
    nums(j) = size(feats{j}, 1);
    fprintf('class %s has %d samples\n', classes{j}, nums(j));
    data = cat(1, data, feats{j});
    assert(size(data,1) == sum(nums));
    feats{j} = [];
    keys = cat(1, keys, key_cell{j});
    index = cat(1, index, index_cell{j});
  end
  fprintf('size data:\n');
  disp(size(data));
  try
    save(file_name, 'data', 'nums', '-v7.3');
    fprintf('Saved ORIDATA to %s\n');
  catch err
    disp(err.identifier);
    error('Save ORIDATA at %s unsucessful\n', file_name);
  end
end
if opts.fuse_hw
  data = fuse_hw(data, opts.fuse_size);
end



function print_num(feats)
for i = 1: length(feats)
  fprintf('%d, ', size(feats{i},1));
end
fprintf('\n');

function names = rep(name, n)
names = cell(n,1);
for i = 1:n
  names{i} = name;
end

