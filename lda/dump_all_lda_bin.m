function dump_all_lda(feat_name, varargin)
ip = inputParser;
ip.addRequired('feat_name',   @isstr);
ip.addParamValue('max_num_per_class', 5500,@isscalar);
ip.addParamValue('do_normalize',      true,@isscalar);
ip.addParamValue('do_lda',      true,@isscalar);
ip.parse(feat_name, varargin{:});
opts = ip.Results;

lda_dir = 'lda/';
lda_data_dir = [lda_dir, 'lda_data/'];
mkdirs({lda_dir, lda_data_dir});
lda_data_dir = [lda_data_dir feat_name '/'];
mkdirs({lda_data_dir});
lda_data_dir = [lda_data_dir int2str(opts.max_num_per_class) '/'];
mkdirs({lda_data_dir});
data_dir = ['lda/data/' feat_name '/' int2str(opts.max_num_per_class) '/'];

trans = load_trans(feat_name, opts);
proj_dim = cell_size_sum(trans,2);
f_start = get_filter_start(trans);

num_class = 20;
matlabpool open
parfor i = 1:num_class
  lda_class(trans, proj_dim, f_start, feat_name, opts, i, lda_data_dir, data_dir); 
end

function lda_class(trans, proj_dim, f_start, feat_name, opts, class_id, dir, data_dir)
in_filename = get_data_filename(data_dir, class_id, opts.do_normalize);
filename = get_data_filename(dir, class_id, opts.do_normalize);

ld = load(in_filename);
proj = lda(ld.data, trans, proj_dim, f_start);
save(filename, 'proj', '-v7.3');
