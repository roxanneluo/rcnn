function compute_all_trans(feat_name, num_filter, class_group, filter_group, varargin)
ip = inputParser;
ip.addRequired('feat_name',   @isstr);
ip.addRequired('num_filter',  @isscalar);
ip.addRequired('class_group', @isscalar);
ip.addRequired('filter_group',@isscalar);
ip.addParamValue('max_num_per_class', 5500,@isscalar);
ip.addParamValue('do_normalize',      true,@isscalar);
ip.addParamValue('dump_interval',     100, @isscalar);
ip.addParamValue('im_start',          1,   @isscalar);
ip.parse(feat_name, num_filter, class_group, filter_group, varargin{:});
opts = ip.Results;
 
lda_dir = './lda/'; data_dir = [lda_dir 'data/']; trans_dir = [lda_dir 'trans/'];
mkdirs({lda_dir, data_dir, trans_dir});
data_dir = [data_dir feat_name '/']; trans_dir = [trans_dir feat_name '/'];
mkdirs({data_dir, trans_dir});
data_dir = [data_dir int2str(opts.max_num_per_class) '/'];
trans_dir  = [trans_dir int2str(opts.max_num_per_class) opts_name(opts) '/'];
mkdirs({data_dir, trans_dir});

fprintf('Dumping Pos\n');
num_cls = 20;
num_cls_batch = ceil(num_cls/class_group);
for i = 1:num_cls_batch
  cls_start = (i-1)*class_group+1;
  cls_end = min(num_cls, cls_start + class_group - 1);
  feat_dim = dump_pos(feat_name, cls_start:cls_end, data_dir, opts);
  fprintf('Pos for classes %d:%d dumped\n', cls_start, cls_end);
end


fprintf('computing trans\n');
dim = feat_dim / num_filter;
num_flt_batch = ceil(num_filter/filter_group);
batch_dim = filter_group * dim;
fprintf('dim=%d,num_filter=%d,feat_dim=%d,filter_group=%d\n',...
    dim, num_filter, feat_dim, filter_group);
trans_err_file = [trans_dir 'trans_err.log'];
trans_err_f = -1;
if opts.do_normalize
  dump_normalize(data_dir, num_cls);
end
for i = 1:num_flt_batch
  batch_start = get_start(i, batch_dim);
  batch_end   = get_end(i, batch_dim, feat_dim);
  fprintf('loading dims %d:%d\n for filter_batch %d\n', ...
      batch_start, batch_end, i);
  [data, nums] = load_dims(data_dir, batch_start, batch_end, num_cls, opts.do_normalize);
  label  = get_label(nums);
  filter_start  = get_start(i, filter_group);
  filter_end    = get_end(i, filter_group, num_filter);
  for f = filter_start:filter_end
    dim_start = get_start(f, dim) - batch_start + 1;
    dim_end   = get_end(f, dim, feat_dim) - batch_start+1;
    trans_file_name = get_trans_filename(trans_dir, f);
    fprintf('Computing trans for filter %d\n', f);
    try
    compute_trans(data(:, dim_start:dim_end), label, num_cls-1, trans_file_name);
    catch err
      if trans_err_f == -1
        trans_err_f = fopen(trans_err_file);
      end
      fprintf('f: %s\n\t%s\n', err.identifier, err.message);
      fprintf(trans_err_f, 'f: %s\n\t%s\n', err.identifier, err.message);
    end
  end
end
if trans_err_f ~= -1
  fclose(trans_err_f);
end

%------------------------------------------------------------------------------
function start = get_start(i, group)
%------------------------------------------------------------------------------
start = (i-1)*group+1;

%------------------------------------------------------------------------------
function tail = get_end(i, group, max_size)
%------------------------------------------------------------------------------
tail = min(i*group, max_size);

%------------------------------------------------------------------------------
function name = opts_name(opts)
%------------------------------------------------------------------------------
name = '';
if opts.do_normalize
  name = [name '_norm'];
end
