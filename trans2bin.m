function trans2bin(feat_name, varargin)
ip = inputParser;
ip.addRequired('feat_name', @isstr);
ip.addParamValue('do_lda', true, @isscalar);
ip.addParamValue('do_normalize', true, @isscalar);
ip.addParamValue('max_num_per_class', 5500, @isscalar);
ip.parse(feat_name, varargin{:});
opts = ip.Results;

filename = get_merged_trans_filename(feat_name, opts.max_num_per_class, ...
    opts, 'bin');
f = fopen(filename, 'w');
trans = load_trans(feat_name, opts);
disp(trans{1}(1:10));
disp(trans{length(trans)}(1:10));
trans_size = size(trans{1});
assert(prod(trans_size) > 0);
for i = 1:length(trans)
  fprintf('writing %d\n', i);
  if ~isempty(trans{i})
    fwrite(f, trans{i}, 'single');
  else
    fwrite(f, zeros(trans_size), 'single');
  end
end
fclose(f);
fprintf('done\n');
