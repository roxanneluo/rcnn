function draw_dot_distr(feat_name, varargin)
ip = inputParser;
ip.addRequired('feat_name', @isstr);
ip.addParamValue('npcls', 5500, @isscalar);
ip.addParamValue('do_normalize', true, @isscalar);
ip.addParamValue('num_sample', 100, @isscalar);
ip.parse(feat_name, varargin{:});
opts = ip.Results;

data_dir = ['lda/data/' feat_name '/' int2str(opts.npcls) '/'];
dot_dir = ['draw-res/dot_distr/']; mkdirs({dot_dir});
dot_dir = [dot_dir feat_name '/']; mkdirs({dot_dir});
dot_dir = [dot_dir int2str(opts.npcls) opts_name(opts) '/']; mkdirs({dot_dir});
sort_dir = ['draw-res/sort/']; mkdirs({sort_dir});
sort_dir = [sort_dir feat_name '/']; mkdirs({sort_dir});
sort_dir = [sort_dir int2str(opts.npcls) opts_name(opts) '/']; mkdirs({sort_dir});

%num_class = 20;
%load_order = [1:14, 16:20, 15];
num_class = 2;
load_order = [6,19];
feat_dim = 512*3*3*384;
mean_file = [dot_dir 'means' opts_name(opts) '.mat'];
if exist(mean_file, 'file')
  fprintf('Loading %s\n', mean_file);
  ld = load(mean_file);
  nums = ld.nums;
  means = ld.means; clear ld;
  means = gpuArray(means); nums = gpuArray(nums);
  ld = load(get_data_filename(data_dir, load_order(end), opts.do_normalize));
  ld.data = gpuArray(ld.data);
else
  gpu_means = gpuArray(zeros(feat_dim, num_class));
  nums = zeros(num_class,1);
  for i = 1:num_class
    clear ld;
    filename = get_data_filename(data_dir, load_order(i), opts.do_normalize);
    fprintf('loading %s.', filename);
    ld = load(filename);
    fprintf('loaded %s.\n', filename);
    ld.data = gpuArray(ld.data);
    gpu_means(:, load_order(i)) = mean(ld.data, 1)';
    nums(i) = size(ld.data,1);
  end
  means = gather(gpu_means);
  save(mean_file, 'means', 'nums', '-v7.3');
  clear means;
  means = gpu_means;
end
nums = gpuArray(nums);
global_IX = gather(get_base_IX((sum(means*diag(nums),2)/sum(nums))'));
mean_IX = gather(sort(means, 2))'; 
mean_norm = sqrt(sum(means.*means));

diary_file = [dot_dir 'mean_std.txt'];
diary(diary_file);
%draw_order = [15, 20:-1:16, 14:-1:1, -1];
class_names = {'aero' 'bike' 'bird' 'boat' 'bottle' 'bus' 'car' 'cat' 'chair' ... 
    'cow' 'table' 'dog' 'horse' 'mbike' 'person' 'plant' 'sheep' 'sofa' ... 
    'train' 'tv' 'background'};
draw_order = [19,6,-1];
for i = 1:num_class
  try
    id = draw_order(i);
    dots = ld.data*means;
    IX = randperm(opts.num_sample, size(ld.data,1));
    fprintf('drawing sort %d(class %d)\n', i, id);
    sample = ld.data(IX,:);
    draw_and_save_sort(gather(sample), sort_dir, ['class-sort-' int2str(id) class_names{id}], mean_IX(id,:));  
    draw_and_save_sort(gather(sample), sort_dir, ['global-sort-' int2str(id) class_names{id}], globalIX);  
    clear sample;
    if ~opts.do_normalize
      data_norm = sqrt(sum(ld.data.*ld.data,2));
      clear ld;
      norm_prod = data_norm*mean_norm;
      norm_prod(norm_prod < eps) = 1;
      angle = acos(dots./norm_prod);
      fprintf('%d:dots mean\t', id); disp(mean(dots));
      fprintf('%d:dots std\t', id); disp(std(dots));
      title = sprintf('dot_distr_%d%s', id, opts_name(opts));
      draw_hist_surf(gather(dots), 100, dot_dir, title);
    else
      clear ld;
      angle = acos(dots);
      data_norm = ones(size(dots,1),1);
    end
    angle = gather(angle); data_norm = gather(data_norm);
    pos_num = sum(dots>0); neg_num = sum(dots<0); ortho_num = sum(dots == 0); num = size(dots,1);
    fprintf('%d:sgn pos_ratio\t', id); disp(pos_num/num);
    fprintf('%d:sgn neg_ratio\t', id); disp(neg_num/num);
    fprintf('%d:sgn rtho_ratio\t', id); disp(ortho_num/num);
    fprintf('%d:mean angle\t', id); disp(mean(angle)/pi*180);
    fprintf('%d:std  angle\t', id); disp(std(angle)/pi*180);
    title = sprintf('angle_distr_%d%s', id, opts_name(opts));
    draw_polar(id, angle, data_norm, dot_dir, title);
    
    if draw_order(i+1) > 0
      filename = get_data_filename(data_dir, draw_order(i+1), opts.do_normalize);
      fprintf('loading %s.', filename);
      ld = load(filename);
      ld.data = gpuArray(ld.data);
      fprintf('loaded %s.\n', filename);
    end
  catch err
    fprintf('%d: %s\n', id, err.identifier);
    fprintf('%d: %s\n', id, err.message);
  end
end

fprintf('done\n');
