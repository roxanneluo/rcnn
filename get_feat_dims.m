% -----------------------------------------------------------------------------
function dims = get_feat_dims(feat_opts)
% -----------------------------------------------------------------------------
dims = zeros(length(feat_opts),1);
for i = 1:length(feat_opts)
  dims(i) = get_feat_part_dim(feat_opts(i));
end

function part_dim = get_feat_part_dim(feat_opt)
  switch(feat_opt.layer)
  case 5
    if feat_opt.d
      part_dim = 256;
    else 
      part_dim = 9216;
    end
  case 6
    part_dim = 4096;
  case 7
    part_dim = 4096;
  case 8
    part_dim = 21;
  otherwise
    assert(false);
  end

