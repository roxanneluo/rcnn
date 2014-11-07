function name = opts_name(opts)
name = '';
if isfield(opts, 'do_normalize') && opts.do_normalize
  name = [name '_norm'];
end
if isfield(opts, 'do_lda') && opts.do_lda
  name = [name '_lda'];
end
