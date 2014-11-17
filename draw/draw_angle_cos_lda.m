function draw_angle_cos_lda(feat_name, class_ids, varargin)
ip = create_input_parser;
ip.addRequired('class_ids', @isvector);
ip.parse(feat_name, class_ids, varargin{:});
opts = ip.Results;

dot_dir = ['draw-res/dot_distr/']; mkdirs({dot_dir});
dot_dir = [dot_dir feat_name '/']; mkdirs({dot_dir});
dot_dir = [dot_dir int2str(opts.max_num_per_class) opts_name(opts) '/']; mkdirs({dot_dir});
cos_dir = [dot_dir 'cos_lda_with_data_norm/'];
angle_dir = [dot_dir 'angle_lda_with_data_norm/'];
mkdirs({cos_dir, angle_dir});

num_class = 20;
mean_norm = get_mean_norm(dot_dir, opts);
[dots, data_norm, class] = get_dots(dot_dir, num_class, opts);
data_cos = get_cos(dots, data_norm, mean_norm);
angle = acos(data_cos);

for i = 1:length(class_ids) 
  id = class_ids(i);
  label = class == id;
  compute_lda_and_draw_scatter(data_cos, data_norm, label, class, num_class, id, cos_dir, 'cos');
  compute_lda_and_draw_scatter(angle, data_norm, label, class, num_class, id, angle_dir, 'angle');
end

%------------------------------------------------------------------------------
function compute_lda_and_draw_scatter(data, data_norm, label, class, num_class, id, dir, name)
%------------------------------------------------------------------------------
trans_filename = [dir 'TRANS_' name '_' int2str(id) '.mat'];
data = cat(2, data, data_norm);
[trans, val] = compute_trans(data, label, 1, trans_filename);
proj = data * trans;
draw_class_scatter(proj, class, id, dir, name, num_class);


%------------------------------------------------------------------------------
function mean_norm = get_mean_norm(dot_dir, opts)
%------------------------------------------------------------------------------
mean_file = [dot_dir 'means' opts_name(opts) '.mat'];
ld = load(mean_file);
mean_norm = sqrt(sum(ld.means.*ld.means));
clear ld;

%------------------------------------------------------------------------------
function [dots, data_norm, class] = get_dots(dot_dir, num_class, opts)
%------------------------------------------------------------------------------
dots = []; data_norm = []; class = [];
for id = 1:num_class
  dot_filename = [dot_dir 'dots_data_norm_' int2str(id) '.mat'];
  ld = load(dot_filename);
  dots = cat(1, dots, ld.dots);
  data_norm = cat(1, data_norm, ld.data_norm);
  class = cat(1, class, ones(size(ld.dots,1), 1)* id);
  clear ld;
  assert(size(dots,1) == size(data_norm,1) && size(data_norm,1) == size(class,1));
end
