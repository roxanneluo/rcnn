function dump_next(feat_name, num_filter, class_group, filter_group, varargin)
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
data_dir = [data_dir int2str(opts.max_num_per_class) '_next/'];
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
