function dump_all_lda(feat_name, imdb_name, varargin)
ip = create_input_parser;
ip.addRequired('imdb_name', @isstr);
ip.addParamValue('read_iter', 1024, @isscalar);
ip.parse(feat_name, imdb_name, varargin{:});
opts = ip.Results;

trans = load_trans(feat_name, opts);
proj_dim = cell_size_sum(trans,2);
filter_start = get_filter_start(trans);

data_dir = ['lda/data/' feat_name '/' int2str(opts.max_num_per_class) '/'];
dump_dir = ['feat_cache/' feat_name opts_name(opts) '/voc_2007_' imdb_name '/'];
assert(exist(dump_dir) == 7);
dump_file = [dump_dir 'gt_pos.mat'];
num_class = 20;
X_pos = cell(num_class,1);
for i = 1:num_class
  class_dump_file = [dump_dir 'gt_pos_' int2str(i) '.mat'];
  if exist(class_dump_file, 'file')
    fprintf('Loading %s\n', class_dump_file);
    ld = load(class_dump_file);
    X_pos{i} = ld.X_pos_class;
    clear ld;
  else
    m = get_matfile(i, data_dir, opts);
    [num, feat_dim] = size(m, 'data');
    [num_batch, batch_size] = get_batch_size(num, opts.read_iter);
    X_pos_class = single(zeros(num, proj_dim));
    for b = 1:num_batch
      num_start = (b-1)*batch_size+1;
      num_end = min(b*batch_size, num);
      fprintf('class%d-batch%d: LDAing (num_start=%d, num_end=%d)\n', i, b,...
          num_start, num_end);
      X_pos_class(num_start:num_end,:) = ... 
        lda(single(m.data(num_start:num_end,:)), trans, proj_dim, filter_start);
    end
    clear m;
    fprintf('%d: saving %s\n', i, class_dump_file);
    save(class_dump_file, 'X_pos_class', '-v7.3');
    X_pos{i} = X_pos_class; clear X_pos_class;
  end
end
fprintf('saving %s\n', dump_file);
save(dump_file, 'X_pos', '-v7.3');

%------------------------------------------------------------------------------
function [num_batch, batch_size] = get_batch_size(num, read_iter) 
%------------------------------------------------------------------------------
batch_size = read_iter;
num_batch = ceil(num/batch_size);

%------------------------------------------------------------------------------
function m = get_matfile(class_id, data_dir, opts)
%-----------------------------------------------------------------------------
filename = get_data_filename(data_dir, class_id, opts.do_normalize);
m = matfile(filename);
