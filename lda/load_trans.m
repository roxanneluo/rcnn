function trans = load_trans(feat_name, opts)
filename = get_merged_trans_filename(feat_name, opts);
ld = load(filename);
trans = ld.trans; clear ld;


