function compute_all_trans_my_write(feat_opt, num_filter, varargin)
ip = inputParser;
ip.addRequired('feat_opt',   @isstruct);
ip.addRequired('num_filter',  @isscalar);
ip.addParamValue('class_group',       20,  @isscalar);
ip.addParamValue('max_num_per_class', 5500,@isscalar);
ip.addParamValue('do_normalize',      true,@isscalar);
ip.addParamValue('dump_interval',     500, @isscalar);
ip.addParamValue('im_start',          1,   @isscalar);
ip.addParamValue('neg_per_im',        0,   @isscalar);
ip.addParamValue('backward_type',     'sb',   @isstr);
ip.parse(feat_opt, num_filter, varargin{:});
opts = ip.Results;
opts.do_lda = true;
 
feat_name = feat_opts_to_string(feat_opt);
lda_dir = './lda/'; 
data_dir = [lda_dir 'data/']; trans_dir = [lda_dir 'trans/']; 
filter_dir = [lda_dir 'filter/'];
mkdirs({lda_dir, data_dir, trans_dir, filter_dir});
data_dir = [data_dir feat_name '/']; trans_dir = [trans_dir feat_name '/'];
filter_dir = [filter_dir feat_name '/'];
mkdirs({data_dir, trans_dir, filter_dir});
data_dir = [data_dir int2str(opts.max_num_per_class) '/'];
trans_dir  = [trans_dir int2str(opts.max_num_per_class) opts_name(opts) '/'];
filter_dir = [filter_dir int2str(opts.max_num_per_class) '/'];
mkdirs({data_dir, trans_dir, filter_dir});
disp(data_dir)

fprintf('Dumping Pos\n');
num_cls = 20;
num_cls_batch = ceil(num_cls/opts.class_group);
for i = 1:num_cls_batch
  cls_start = (i-1)*opts.class_group+1;
  cls_end = min(num_cls, cls_start + opts.class_group - 1);
  feat_dim = dump_pos_online(feat_opt, cls_start:cls_end, data_dir, opts);
  fprintf('Pos for classes %d:%d dumped\n', cls_start, cls_end);
end

%TODO only handles normalized version
fprintf('Norm and Cut\n');
nums = norm_and_cut(data_dir, num_cls, filter_dir, num_filter, ...
    opts.do_normalize, feat_dim, opts);

fprintf('Computing trans\n');
dim = feat_dim / num_filter;
trans_err_file = [trans_dir 'trans_err.log'];
trans_err_f = -1;
err_messages = cell(num_filter, 1);
label = get_label(nums);
for f = 1:num_filter
  err_messages{f} = compute_and_dump_trans(f, filter_dir, label, trans_dir, ...
      dim, num_cls, opts);
end
for f = 1:num_filter
  if ~isempty(err_messages{f})
    if trans_err_f < 0
      trans_err_f = fopen(trans_err_file, 'w');
    end
    fprintf(trans_err_f, err_messages{f});
  end
end
if trans_err_f ~= -1
  fclose(trans_err_f);
end
%------------------------------------------------------------------------------
function err_msg = compute_and_dump_trans(f, filter_dir, label, trans_dir, ...
    dim, num_class, opts)
%------------------------------------------------------------------------------
err_msg = '';
filename = get_data_filename(filter_dir, f, opts.do_normalize);
data = my_read(filename, [inf, dim]);
trans_filename = get_trans_filename(trans_dir, f);
fprintf('Computing trans for filter %d\n', f);
try
  compute_trans(data, label, num_class-1, trans_filename);
catch err
  err_msg = sprintf('[ERROR] %d: %s\n\t%s\n', f, err.identifier, err.message);
  fprintf(err_msg);
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
%{
o
%}
