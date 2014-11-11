function draw_dot_distr(feat_name, varargin)
ip = inputParser;
ip.addRequired('feat_name', @isstr);
ip.addParamValue('npcls', 5500, @isscalar);
ip.addParamValue('do_normalize', true, @isscalar);
ip.parse(feat_name, varargin{:});
opts = ip.Results;

data_dir = ['lda/data/' feat_name '/' int2str(opts.npcls) '/'];
dot_dir = ['draw-res/dot_distr/']; mkdirs({dot_dir});
dot_dir = [dot_dir feat_name '/']; mkdirs({dot_dir});
dot_dir = [dot_dir int2str(opts.npcls) '/']; mkdirs({dot_dir});
dot_dir = [dot_dir opts_name(opts) '/']; mkdirs({dot_dir});

%num_class = 20;
%load_order = [1:14, 16:20, 15];
num_class = 2;
load_order = [6,19];
feat_dim = 512*3*3*384;
mean_file = [dot_dir 'means' opts_name(opts) '.mat'];
if exist(mean_file, 'file')
  ld = load(mean_file);
  means = ld.means; clear ld;
else
  means = zeros(feat_dim, num_class);
  for i = 1:num_class
    clear ld;
    filename = get_data_filename(data_dir, load_order(i), opts.do_normalize);
    fprintf('loading %s.', filename);
    ld = load(filename);
    fprintf('loaded %s.\n', filename);
    means(:, load_order(i)) = mean(ld.data, 1)';
  end
  save(mean_file, 'means', '-v7.3');
end

mean_norm = sqrt(sum(means.*means));

diary_file = [dot_dir 'mean_std.txt'];
diary(diary_file);
%draw_order = [15, 20:-1:16, 14:-1:1, -1];
draw_order = [19,6];
for i = 1:num_class
  try
    id = draw_order(i);
    dots = ld.data*means;
    if ~opts.do_normalize
      data_norm = sqrt(sum(ld.data.*ld.data,2));
      clear ld;
      norm_prod = data_norm*mean_norm;
      norm_prod(norm_prod < eps) = 1;
      angle = acos(dots./norm_prod);
      fprintf('%d:dots mean\t', id); disp(mean(dots));
      fprintf('%d:dots std\t', id);disp(std(dots));
      title = sprintf('dot_distr_%d%s', id, opts_name(opts));
      draw_hist_surf(dots, 100, dot_dir, title);
    else
      clear ld;
      angle = acos(dots);
      data_norm = ones(size(dots,1),1);
    end
    pos_num = sum(dots>0); neg_num = sum(dots<0); ortho_num = sum(dots == 0); num = size(dots,1);
    fprintf('%d:sgn pos_ratio\t', id); disp(pos_num/num);
    fprintf('%d:sgn neg_ratio\t', id); disp(neg_num/num);
    fprintf('%d:sgn rtho_ratio\t', id); disp(ortho_num/num);
    fprintf('%d:mean angle\t', id); disp(mean(angle)/pi*180);
    fprintf('%d:std  angle\t', id); disp(std(angle)/pi*180);
    title = sprintf('angle_distr_%d_%s', id, opts_name(opts));
    draw_polar(id, angle, data_norm, dot_dir, title);
    
    if draw_order(i+1) > 0
      filename = get_data_filename(data_dir, draw_order(i+1), opts.do_normalize);
      fprintf('loading %s.', filename);
      ld = load(filename);
      fprintf('loaded %s.\n', filename);
    end
  catch err
    fprintf('%d: %s\n', id, err.identifier);
    fprintf('%d: %s\n', id, err.message);
  end
end

fprintf('done\n');
