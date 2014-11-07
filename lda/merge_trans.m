function merge_trans(feat_name, num_filter, npcls, do_normalize)
opts = []; opts.do_normalize = do_normalize;
trans_dir = ['./lda/trans/' feat_name '/'];
trans_in_dir = [trans_dir int2str(npcls) opts_name(opts) '/'];
merge_file = get_merged_trans_filename(feat_name, npcls, opts);
merge_err_file = sprintf('%s%d_TRANS_ERR.log', trans_dir, npcls);
merge_err_f = -1;
trans = cell(num_filter);
for i = 1:num_filter
  filename = get_trans_filename(trans_in_dir, i);
  if ~exist(filename, 'file')
    if merge_err_f == -1
      merge_err_f = fopen(merge_err_f, 'w');
    end
    fprintf(merge_err_f, '%d: %s does not exist\n', i, filename);
    fprintf('%d: %s does not exist\n', i, filename);
    trans{i} = [];
  else
    fprintf('Loading %s\n', filename);
    ld = load(filename);
    trans{i} = ld.trans;
    clear ld;
  end
end
fprintf('saving %s\n', merge_file);
save(merge_file, 'trans');
fprintf('Saved\n');
if merge_err_f ~= -1
  fclose(merge_err_f);
end
