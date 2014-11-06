function pca_analysis(feat_name, varargin)
ip = inputParser;
ip.addRequired('feat_name', @isstr);
ip.addParamValue('max_num_bg_check',   65536, @isscalar);
ip.addParamValue('max_num_bg',         10000, @isscalar);
ip.addParamValue('max_num_per_class',     -1, @isscalar);
ip.addParamValue('proj_dim',               0, @isscalar);
ip.addParamValue('pca_ratio',              1, @isscalar);
ip.addParamValue('whiten',              true, @isscalar);
ip.addParamValue('draw_dim',           false, @isscalar);
ip.addParamValue('draw_eigen_value',   true, @isscalar);
ip.addParamValue('draw_original',       true, @isscalar);
ip.addParamValue('draw_proj',           true, @isscalar);
ip.addParamValue('draw_2d_proj',        true, @isscalar);
ip.parse(feat_name, varargin{:});
opts = ip.Results;

pca_dir = 'draw-res/pca/';
feat_dir = [pca_dir feat_name '/'];
dir = [feat_dir '/' int2str(opts.proj_dim) '/'];
mkdirs({pca_dir, feat_dir, dir});

VOCdevkit = './datasets/VOCdevkit2007';
imdb = imdb_from_voc(VOCdevkit, 'trainval', '2007');
data_file = [feat_dir 'data_numpcls5000.mat'];
%data_file = [dir 'data_projdim' int2str(opts.proj_dim) ...
%  '_numpcls' int2str(opts.max_num_per_class) '.mat'];
if exist(data_file, 'file')
  fprintf('Load data from file %s\n', data_file);
  ld = load(data_file);
  data = ld.data; 
  nums = ld.nums;
  dims = ld.dims;
  mean_norm = ld.mean_norm;
  equal_dim = ld.equal_dim;
  clear ld;
  if 0 < opts.max_num_per_class && opts.max_num_per_class < 5000
    [data, nums] = filter(data, nums, opts);
    size(data)
    nums
  end
else
  fprintf('preparing data\n');
  [data, nums, dims, mean_norm, equal_dim] = prepare_data(feat_name, imdb, opts);
  save(data_file, 'data', 'nums', 'dims', 'mean_norm','equal_dim');
end

% compute and draw eigen vectors and eigen values
fprintf('calculating eigen values and eigen vectors\n');
%eigen_file = [dir 'eigen_projdim' int2str(opts.proj_dim) '_numpcls'...
 % int2str(opts.max_num_per_class) '.mat'];
eigen_file = [feat_dir 'eigen_projdim' int2str(opts.proj_dim) '_numpcls'...
  '5000.mat'];
if exist(eigen_file)
  ld = load(eigen_file);
  eigen_vec = ld.eigen_vec;
  eigen_val = ld.eigen_val; clear ld;
  if opts.pca_ratio < 1
    [eigen_vec, eigen_val] = eigen_by_ratio(eigen_vec, eigen_val, opts.pca_ratio);
    dir = sprintf('%s/pcaRatio%f/', dir, opts.pca_ratio);
  end
else
  options = [];
  options.ReducedDim = opts.proj_dim;
  if opts.pca_ratio ~= 1
    options.PCARatio = pca_ratio;
  end
  [eigen_vec, eigen_val] = PCA(data, options);
  eigen_vec = real(eigen_vec); % why eigen_vec has imaginary part
  scale = struct('mean_norm', mean_norm, 'dims', dims, 'feat_dim', sum(dim), ...
      'equal_dim', equal_dim);
  save(eigen_file, 'eigen_val', 'eigen_vec', 'scale');
end
fprintf('size of eigen vec = [%d, %d]\n', size(eigen_vec, 1), size(eigen_vec,2));
fprintf('size of eigen val = [%d, %d]\n', size(eigen_val, 1), size(eigen_val,2));

if opts.draw_eigen_value
  fprintf('ploting eigen values\n');
  draw_eigen_value(imdb, dir, eigen_val);
end

if opts.draw_original
  fprintf('drawing histogram of the original data\n');
  ori_dir = [dir 'original/'];
  mkdirs({ori_dir});
  draw_all_histogram(data, nums, imdb, ori_dir, 'original', opts.draw_dim);
end

if opts.draw_proj || opts.draw_2d_proj
  proj_feat = [];
  for i = 1:length(nums)
    num = nums(i);
    proj_feat = cat(1, proj_feat, project(data(1:num, :), eigen_vec, eigen_val, opts.whiten));
    data(1:num,:) = [];
    fprintf('size data = [%d, %d]\n', size(data,1), size(data,2));
  end
end

suf = 'proj';
if opts.whiten 
  suf = [suf, '_w'];
end
if opts.draw_proj
  fprintf('drawing histogram of the projected data\n');
  proj_dir = [dir 'proj/'];
  mkdirs({proj_dir});
  draw_all_histogram(proj_feat, nums, imdb, proj_dir, suf, opts.draw_dim);
end
if opts.draw_2d_proj
  fprintf('drawing projected data on 2d plane\n');
  draw_2d_proj(imdb, proj_feat, nums, dir, suf);
end

%-------------------------------------------------------
function [data, nums, dims, mean_norm, equal_dim] = prepare_data(feat_name, imdb, opts)
%-------------------------------------------------------
%so far all classes are done together
model_file = ['cachedir/' feat_name '/voc_2007_trainval/rcnn_model.mat'];
rcnn_model = rcnn_load_model(model_file);
rcnn_model = add_model_fields(rcnn_model);
rcnn_model.norm_weight = true;
rcnn_model.proj = false;
dims = rcnn_model.dims;
mean_norm = rcnn_model.training_opts.feat_norm_mean;

class_ids = [imdb.class_ids, length(imdb.class_ids)+1];
classes = {imdb.classes{:}, 'background'};
cache_name = rcnn_model.training_opts.cache_name;
feats = cell(length(class_ids), 1);
neg_ovr_thresh = 0.3;
exceed = zeros(length(class_ids),1);
max_check = load('max_check_num.txt');
if opts.max_num_per_class > 0
  max_check = min(opts.max_num_per_class, max_check);
end
max_check = [max_check; opts.max_num_bg_check];
for i = 1:length(imdb.image_ids)
  if all(exceed)
    fprintf('all classes reaches max_num_per_class = %d, %d\n', ...
        opts.max_num_per_class, opts.max_num_bg_check);
    break;
  end

  tic_toc_print('%s: features %d/%d\n', ...
                procid(), i, length(imdb.image_ids));
  d = rcnn_load_cached_pool5_features(cache_name, ...
      imdb.name, imdb.image_ids{i}, rcnn_model.exist_r, {'class'; 'overlap'});
  feat = get_feature(d.feat, rcnn_model, imdb.name, ...
      struct('image_id', imdb.image_ids{i}, 'IX', []), []);
  feat = rcnn_scale_features(feat, mean_norm, ...
      rcnn_model);

  for j = class_ids
    if exceed(j)
      continue;
    end
    if isempty(feats{j})
      feats{j} = single([]);
    end
    if j == length(class_ids)
      sel = find(d.class == 0  & all(d.overlap < neg_ovr_thresh, 2)); 
    else
      sel = find(d.class == j);
    end
    if ~isempty(sel)
      feats{j} = cat(1, feats{j}, feat(sel,:));
      if max_check(j) >= 0 && size(feats{j},1) >= max_check(j)
        exceed(j) = true;
        feats{j} = feats{j}(1:max_check(j),:);
        fprintf('class %d-%s exceeds max_check=%d\n', i, classes{j}, max_check(j));
      end
    end
  end
end
data = [];
nums = zeros(length(class_ids),1);
for j = class_ids 
  if j == max(class_ids) && opts.max_num_bg_check >=0 && opts.max_num_bg >= 0
    IX = randperm(opts.max_num_bg_check, opts.max_num_bg);
    feats{j} = feats{j}(IX,:);
  end
  nums(j) = size(feats{j}, 1);
  fprintf('class %s has %d samples\n', classes{j}, nums(j));
  data = cat(1, data, feats{j});
  assert(size(data,1) == sum(nums));
  feats{j} = [];
end

%-------------------------------------------------------
function [eigen_vec, eigen_val] = eigen_by_ratio(eigen_vec, eigen_val, pca_ratio);
%-------------------------------------------------------
s = sum(eigen_val);
p_sum = 0;
proj_dim = length(eigen_val);
for i = 1:length(eigen_val)
  p_sum = p_sum + eigen_val(i);
  if p_sum >= pca_ratio*s
    proj_dim = i; 
    fprintf('!!!!!!!!!proj_dim = %d, pca_ratio=%f, p_sum = %f!!!!!!!!!!!\n', proj_dim, pca_ratio,p_sum);
    break;
  end
end
eigen_val = eigen_val(1:proj_dim);
eigen_vec = eigen_vec(:,1:proj_dim);

%-------------------------------------------------------
function  [data, nums] = filter(data, nums, opts)
%-------------------------------------------------------
num_bg = opts.max_num_bg;
if num_bg <= 0
  num_bg = inf;
end
num_start = 1;
num_pick= load('max_check_num.txt'); 
num_pick = min(num_pick, opts.max_num_per_class);
num_pick = [num_pick; num_bg];
num_pick = min(nums, num_pick);

num_start = 0;
for i = 1:length(nums)
  if num_pick(i) < nums(i)
    IX = randperm(nums(i), nums(i) - num_pick(i));
    data(num_start+IX,:) = [];
    nums(i) = num_pick(i);
  end
  num_start = num_start + nums(i);
end


%-------------------------------------------------------
function draw_eigen_value(imdb, dir, eigen_val);
%-------------------------------------------------------
f = figure();
plot(1:length(eigen_val), eigen_val);
title_str = 'eigen_value';
title(title_str);
saveas(gcf, [dir title_str '.jpg']);
close(f);

%-------------------------------------------------------
function  draw_2d_proj(imdb, feat, nums, dir, suf);
%-------------------------------------------------------
f = figure();
cmap = jet(length(nums));
num_start = 1;
for i = 1:length(nums)
  num_end = num_start+nums(i)-1;
  scatter(feat(num_start:num_end,1), feat(num_start:num_end,2),[],cmap(i,:));
  hold on
  num_start = num_end+1;
end
length(nums)
length(imdb.classes)
legend(imdb.classes{:}, 'background');
title_str = ['2d-proj-' suf];
title(title_str);
saveas(gcf, [dir title_str '.jpg']);
close(f);

%-------------------------------------------------------
function draw_all_histogram(data, nums, imdb, dir, suf, draw_dim)
%-------------------------------------------------------
dir = [dir 'hist/'];
mkdirs({dir});

num_bin = 100;
num_dim = size(data,2);
[counts, centers] = draw_hist_surf(data, num_bin, dir, ['all-' suf], jet);
% TODO actually I can plot a hist surf and hist of each dim for both all and each class

if draw_dim
  fprintf('draw_dim\n');
  for i = 1:num_dim
    f = figure();
    title_str = sprintf('dim%d-%s', i, suf);
    fprintf('ploting %s\n', title_str);
    title(title_str);
    plot(centers, counts(:,i));
    saveas(gcf, [dir title_str '.jpg']);
    close(f);
  end
end
cmap = jet;
num_start = 1;
num_class = length(imdb.class_ids)+1;
nums
fprintf('draw each class');
for i = 1:num_class
  if i == num_class
    class_name = 'background';
  else
    class_name = imdb.classes{i};
  end
  num_end = num_start+nums(i)-1;
  draw_hist_surf(data(num_start:num_end,:), num_bin, dir, [class_name '-' suf], cmap);
  num_start = num_end+1;
end

%-------------------------------------------------------
function [counts, centers] = draw_hist_surf(data, num_bin, dir, suf, cmap)
%-------------------------------------------------------
fprintf('drawing %s num_bin = %d\n', suf, num_bin);
[counts, centers] = hist(data, num_bin);
fprintf('counts and centers calculated\n');
f = figure();
num_dim = size(data,2);
[X, Y] = meshgrid(1:num_dim, centers) ;
fprintf('meshgrided');
surf(X,Y,counts); colormap(cmap);
fprintf('surf drawn');
title(suf);
view(2);
fprintf('saving at %s%s.jpg\n', dir, suf);
saveas(gcf,[dir suf '.jpg']);
saveas(gcf,[dir suf '.fig']);
fprintf('saved\n');
close(f);

%-------------------------------------------------------


