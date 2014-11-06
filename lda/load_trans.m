function trans = load_trans(feat_name, max_num_per_class)
if ~exist(max_num_per_class)
  max_num_per_class = 200;
end
filename = sprintf('./lda/trans/%s/%d-TRANS.mat', feat_name, max_num_per_class);
ld = load(filename);
trasn = ld.trans; clear ld;

