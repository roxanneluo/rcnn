function lda(feat_name, num_filter, varargin)
ip = inputParser;
ip.addRequired('feat_name', @isstr);
ip.addRequired('num_filter', @isscalar);
ip.addParamValue('max_num_bg_check',   65536, @isscalar);
ip.addParamValue('max_num_bg',          5000, @isscalar);
ip.addParamValue('max_num_per_class',   5000, @isscalar);
ip.addParamValue('do_softmax',         false, @isscalar);
ip.parse(feat_name, num_filter, varargin{:});
opts = ip.Results;
pre = sprintf('npc%d_nbg%d%s', opts.max_num_per_class, opts.max_num_bg, sfname(opts.do_softmax));
lda_dir = ['draw-res/lda/'];
dir =  [lda_dir feat_name '/'];
mkdirs({lda_dir, dir});

[data, nums] = prepare_data(feat_name, opts, dir);
labels = cell(2,1);
labels{1} = all_labels(data, nums);
labels{2} = label_each(data, nums);
%data_funcs = {@whole, @dim_share, @dim_each};
data_funcs = {@dim_share, @dim_each};
lda_funcs = {@lda_all_labels, @lda_label_each};
lda_names = {'all_labels', 'label_each'};
for i = 1:length(data_funcs)
  data_func = data_funcs{i};
  data_func(data, nums, labels, lda_funcs, lda_names, num_filter, pre, dir, opts.do_softmax);
end


%------------------------------------------------------------------------------
function label = label_each(data, nums)
%------------------------------------------------------------------------------
num_class = 20;
num = size(data,1);
label = zeros(num, num_class);
num_start = 1;
for i = 1:num_class
  num_end = num_start+nums(i)-1;
  label(num_start:num_end,i) = 1;
  num_start = num_end+1;
end

%------------------------------------------------------------------------------
function label = all_labels(data, nums)
%------------------------------------------------------------------------------
label = zeros(size(data,1),1);
num_start = 1;
for i = 1:length(nums)
  num_end = num_start + nums(i) - 1;
  label(num_start: num_end) = i;
  num_start = num_end + 1;
end

%------------------------------------------------------------------------------
function dim_each(data, nums, labels, lda_funcs, lda_names, num_filter, pre, dir, do_softmax)
%------------------------------------------------------------------------------
data_name = 'dim_each';
pre = sprintf('%s-%s', pre, data_name);
num_class = 20;

for i = 1:length(lda_funcs)
  lda_func = lda_funcs{i};
  lda_name = lda_names{i};
  name = [pre, '-', lda_names{i}];
  data_file_name = [dir 'DATA-' name '.mat'];

  try
    ld = load(data_file_name);
    proj = ld.proj; clear ld;
    fprintf('Loaded data from %s\n', data_file_name)
  catch
    [num, feat_dim] = size(data);
    dim = feat_dim / num_filter;
    proj = zeros(num, num_filter*num_class);
    for j = 0:num_filter-1
      jstart = j*dim+1;
      jend = (j+1)*dim;
      trans_file_name = sprintf('%sTRANS%d-%s.mat', dir, j, name);

     % proj(:, j*num_class+1:(j+1)*num_class) = ...
      haha = lda_func(data(:, jstart:jend), nums, labels{i}, trans_file_name, do_softmax);
      size(haha)
      proj(:, j*num_class+1:(j+1)*num_class) = haha; 
    end
  end
  draw_sort(proj, nums, name, dir);
end


%------------------------------------------------------------------------------
function dim_share(data, nums, labels, lda_funcs, lda_names, num_filter, pre, dir, do_softmax)
%------------------------------------------------------------------------------
data_name = 'dim_share';
pre = sprintf('%s-%s', pre, data_name);
class_num = 20;

[num, feat_dim] = size(data);
dim = feat_dim/num_filter;
new_data = zeros(num_filter*num, dim);
for i = 0:num_filter-1
  jstart = i*dim+1;
  jend = (i+1)*dim;
  istart = i*num+1;
  iend = (i+1)*num;
  new_data(istart:iend,:) = data(:, jstart:jend);
end
clear data;
for i = 1:length(labels)
  labels{i} = repmat(labels{i}, [num_filter, 1]);
end

for i = 1:length(lda_funcs)
  name = [pre '-' lda_names{i}];
  data_file_name = [dir 'DATA-' name '.mat'];
  trans_file_name = [dir 'TRANS-' name '.mat'];
  try 
    ld = load(data_file_name);
    proj = ld.proj; clear ld;
  catch
    fprintf('~~~~~~~~~~~~Doing %s ~~~~~~~~~~~\n', name);
    lda_func = lda_funcs{i};
    Z = lda_func(new_data, nums, labels{i}, trans_file_name, do_softmax);
    proj = zeros(num, num_filter*class_num);
    for i = 0:num_filter-1
      istart = i*num+1;
      iend = (i+1)*num;
      jstart = i*class_num+1;
      jend = (i+1)*class_num;
      proj(:,jstart:jend) = Z(istart:iend, :);
    end
    clear Z;
    draw_sort(proj, nums, name, dir);
  end
end


%------------------------------------------------------------------------------
function str = sfname(do_softmax)
%------------------------------------------------------------------------------
str = '';
if do_softmax
  str = '-sf';
end

%------------------------------------------------------------------------------
function whole(data, nums, labels, lda_funcs, lda_names, num_filter, pre, dir, do_softmax)
%------------------------------------------------------------------------------
data_name = 'whole';
pre = sprintf('%s-%s', pre, data_name);
for i = 1:length(lda_funcs)
  name = [pre, '-', lda_names{i}];
  data_file_name = [dir 'DATA-' name '.mat'];
  try
    ld = load(file_name);
    proj = ld.proj; clear ld;
    fprintf('loaded proj from %s\n', data_file_name);
  catch
    fprintf('~~~~~~~~~~~~Doing %s ~~~~~~~~~~~\n', name);
    trans_file_name = [dir 'TRANS-' name '.mat'];
    lda_func = lda_funcs{i};
    proj = lda_func(data, nums, labels{i}, trans_file_name, do_softmax);
    fprintf('Saving at %s\n', data_file_name);
    save(data_file_name, 'proj');
  end
  draw_sort(proj, nums, name, dir);
end


%------------------------------------------------------------------------------
function proj = lda_all_labels(data, nums, label, file_name, do_softmax)
%------------------------------------------------------------------------------
try
  ld = load(file_name);
  trans = ld.trans; clear ld;
  fprintf('Loaded trans from %s\n', file_name);
catch
  fprintf('Calculating FDA for %s\n', file_name);
  [proj, trans] = FDA(data', label);
  save(file_name, 'trans');
end
proj = data*trans;
if do_softmax
  proj = softmax(proj);
end

%------------------------------------------------------------------------------
function proj = lda_label_each(data, nums, label, file_name, do_softmax)
%------------------------------------------------------------------------------
num_class = 20;
assert(num_class == size(label, 2));
[num, dim] = size(data);

try
  ld = load(file_name);
  trans = ld.trans; clear ld;
  fprintf('Loaded trans from %s\n', file_name);
catch
  fprintf('Calculating FDA for %s\n', file_name);
  trans = zeros(dim, num_class);
  for i = 1:num_class
    [Z , trans(:,i)] = FDA(data', label(:,i));
    proj(:,i) = Z';
  end
  save(file_name, 'trans');
end
proj = data*trans;
if do_softmax
  proj = softmax(proj);
end
