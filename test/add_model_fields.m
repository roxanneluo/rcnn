function model = add_model_fields(model)
global equal_dim my_test_feat_opts norm_weight svm_C
feat_opts = model.feat_opts;
my_test_feat_opts = feat_opts;
equal_dim = true;
norm_weight = true;
svm_C = 10^-3;
global proj whiten pca_ratio
proj = false; whiten = false; pca_ratio = 1;

model = set_field(model, 'dims',        get_feat_dims(feat_opts));
model = set_field(model, 'exist_r',     exist_response(feat_opts));
model = set_field(model, 'exist_w',     exist_weight(feat_opts));
model = set_field(model, 'equal_dim',   equal_dim);
model = set_field(model, 'norm_weight', norm_weight);
model = set_field(model, 'proj',        proj);
model = set_field(model, 'whiten',      whiten);
model = set_field(model, 'pca_ratio',   pca_ratio);
