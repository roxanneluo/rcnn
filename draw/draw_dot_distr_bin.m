function draw_dot_distr_bin(feat_name, varargin)
ip = create_input_parser;
ip.addParamValue('read_iter', 1024, @isscalar);
ip.addParamValue('num_draw', 1024, @isscalar);
ip.parse(feat_name, varargin{:});
opts = ip.Results;

data_dir = ['lda/data/' feat_name '/' int2str(opts.max_num_per_class) '/'];
dot_dir = ['draw-res/dot_distr/']; mkdirs({dot_dir});
dot_dir = [dot_dir feat_name '/']; mkdirs({dot_dir});
dot_dir = [dot_dir int2str(opts.max_num_per_class) opts_name(opts) '_with_norm_mean/']; mkdirs({dot_dir});
sort_dir = ['draw-res/sort/']; mkdirs({sort_dir});
sort_dir = [sort_dir feat_name '/']; mkdirs({sort_dir});
sort_dir = [sort_dir int2str(opts.max_num_per_class) opts_name(opts) '/']; mkdirs({sort_dir});

num_class = 20;
load_order = [1:6, 8:20, 7];
feat_dim = 512*3*3*384;
mean_file = [dot_dir 'means' opts_name(opts) '.mat'];
nums = load('num_pos.txt');
disp(nums');
if exist(mean_file, 'file')
  fprintf('Loading %s\n', mean_file);
  ld = load(mean_file);
  means = ld.means;
  clear ld;
else
  means = single(zeros(feat_dim, num_class));
  for i = 1:num_class
    id = load_order(i);
    filename = [dot_dir 'm_' int2str(id) opts_name(opts) '.mat'];
    if exist(filename, 'file')
      ld = load(filename);
      means(:,id) = ld.m; clear ld;
    else
      m = compute_mean(id, data_dir, nums(id), feat_dim, opts);
      save(filename, 'm', '-v7.3');
      means(:,id) = m; clear m;
    end
  end
  save(mean_file, 'means', '-v7.3');
end
mean_norm = sqrt(sum(means.*means));


diary_file = [dot_dir 'mean_std.txt'];
diary(diary_file);
draw_order = [7, 20:-1:8, 6:-1:1];
class_names = {'aero' 'bike' 'bird' 'boat' 'bottle' 'bus' 'car' 'cat' 'chair' ... 
    'cow' 'table' 'dog' 'horse' 'mbike' 'person' 'plant' 'sheep' 'sofa' ... 
    'train' 'tv' 'background'};
for i = 1:num_class
  id = draw_order(i);
  dot_filename = [dot_dir 'dots_data_norm_' int2str(id) '.mat'];
  if exist(dot_filename, 'file')
    fprintf('loading %s\n', dot_filename);
    ld = load(dot_filename); dots = ld.dots; data_norm = ld.data_norm;
    clear ld;
  else
    [dots, data_norm] = compute_dots(id, data_dir, min(nums(i), opts.num_draw),...
                                    means, feat_dim, opts);
    save(dot_filename, 'dots', 'data_norm', '-v7.3');
  end
  if ~opts.do_normalize
    disp_mean_std(id, dots, 'dots');
    title = sprintf('dot_distr_%d%s', id, opts_name(opts));
    draw_hist_surf(dots, 100, dot_dir, title);
    
    data_cos = get_cos(dots, data_norm, mean_norm);
  else 
    data_cos = dots;
  end
  clear dots;
  disp_sgn(id, data_cos);
  angle = acos(data_cos); clear data_cos;
  disp_mean_std(id, angle/pi*180, 'angle');
  title = sprintf('angle_polar_%d%s', id, opts_name(opts));
  draw_polar(id, angle, data_norm, dot_dir, title);
  title = sprintf('angle_distr_%d%s', id, opts_name(opts));
  draw_hist_surf(angle/pi*180, 100, dot_dir, title);
end


%------------------------------------------------------------------------------
function [dots, data_norm] = compute_dots(class_id, data_dir, num, ...
                                          means, feat_dim, opts)
%------------------------------------------------------------------------------
data_filename = get_data_filename(data_dir, class_id, opts.do_normalize);
batch_size = opts.read_iter; num_batch = ceil(num/batch_size); 
num_left = num;
dots = []; data_norm = [];
f = fopen(data_filename, 'r');
for i = 1:num_batch
  data = fread(f, [feat_dim, min(num_left, batch_size)], 'single')';
  num_left = num_left - size(data,1);
  fprintf('%d: read %s [%d, %d], num_left=%d\n', class_id,  data_filename, ...
          size(data, 1), size(data,2), num_left);
  dots = cat(1, dots, data * means);
  data_norm = cat(1, data_norm, sqrt(sum(data.*data, 2)));
  clear data;
end
fclose(f);

%------------------------------------------------------------------------------
function disp_mean_std(id, data, name)
%------------------------------------------------------------------------------
fprintf('%d: %s mean\t', id, name); disp(mean(data));
fprintf('%d: %s std\t', id, name); disp(std(data));

%------------------------------------------------------------------------------
function data_cos = get_cos(dots, data_norm, mean_norm) 
%------------------------------------------------------------------------------
norm_prod = data_norm*mean_norm;
norm_prod(norm_prod < eps) = 1;
data_cos = dots./norm_prod;


%------------------------------------------------------------------------------
function disp_sgn(id, data_cos)
%------------------------------------------------------------------------------
pos_num = sum(data_cos>eps); neg_num = sum(data_cos<-eps); 
ortho_num = sum(abs(data_cos) <= eps); num = size(data_cos,1);
fprintf('%d:sgn pos_ratio\t', id); disp(pos_num/num);
fprintf('%d:sgn neg_ratio\t', id); disp(neg_num/num);
fprintf('%d:sgn rtho_ratio\t', id); disp(ortho_num/num);


%------------------------------------------------------------------------------
function m = compute_mean(class_id, data_dir, num, feat_dim, opts)
%------------------------------------------------------------------------------
data_filename = get_data_filename(data_dir, class_id, opts.do_normalize);
num_batch = ceil(num/opts.read_iter); batch_size = opts.read_iter;
m = single(zeros(feat_dim, 1));
cnt = 0;
f = fopen(data_filename, 'r');
for i = 1:num_batch
  data = fread(f, [feat_dim, batch_size], 'single');
  cnt = cnt + size(data,2);
  fprintf('read %s [%d, %d], cnt=%d\n', data_filename, ...
    size(data,1), size(data,2), cnt);
  m = m+sum(data, 2); clear data;
end
fclose(f);
assert(cnt == num);
m = m/num;
