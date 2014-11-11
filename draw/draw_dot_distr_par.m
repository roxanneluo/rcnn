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

num_class = 2;
%load_order = [1:14, 16:20, 15];
load_order = [1,2];
feat_dim = 512*3*3*384;
means = zeros(feat_dim, num_class);
matlabpool open
parfor i = 1:num_class
  means(:, i) = get_mean(i, data_dir, opts);
end

diary_file = [dot_dir 'mean_std.txt'];
diary(diary_file);
draw_order = [2,1,-1];
%draw_order = [15, 20:-1:16, 14:-1:1, -1];
parfor i = 1:num_class
  draw_class_hist(draw_order(i), means, dot_dir, data_dir, opts)
end

fprintf('done\n');

%------------------------------------------------------------------------------
function m = get_mean(id, data_dir, opts)
%------------------------------------------------------------------------------
filename = get_data_filename(data_dir, id, opts.do_normalize);
fprintf('loading %s.', filename);
ld = load(filename);
fprintf('loaded %s.\n', filename);
m = mean(ld.data, 1)';

%------------------------------------------------------------------------------
function draw_class_hist(id, means, dot_dir, data_dir, opts)
%------------------------------------------------------------------------------
filename = get_data_filename(data_dir, id, opts.do_normalize);
fprintf('loading %s.', filename);
ld = load(filename);
fprintf('loaded %s.\n', filename);

dots = ld.data*means;
clear ld;
title = sprintf('dot_distr_%d%s', id, opts_name(opts));
draw_hist_surf(dots, 100, dot_dir, title);
fprintf('%d:mean\t', id); disp(mean(dots));
fprintf('%d:std\t', id);disp(std(dots));
sgn = (dots>0)-(dots<0);
fprintf('%d:mean sgn\t', id); disp(mean(sgn));
fprintf('%d:std sgn\t', id);disp(std(sgn));

  
