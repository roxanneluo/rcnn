function create_feat_opts()
% grad+act x each layer
  for i = 1:test_num_layer
    cnt = cnt+1;
    cell_feat_opts{cnt} = struct('layer', i+4, 'b_or_d', {'blob', 'diff'},...
        'w_or_r', {'response', w_or_r}, 'combine', combine);
  end
  % fc7 + (grad/act x pool5/fc6)
  for i = 1:test_num_layer-1
    layer = i+4;
    for bd = 1:2
      b_or_d = b_or_ds{bd};
      cnt = cnt+1;
      cell_feat_opts(cnt) = struct('layer', {7, layer}, 'b_or_d', {'blob', b_or_d},...
          'w_or_r', w_or_r, 'combine', combine);
    end
  end
  cnt = cnt+1;
  cell_feat_opts(cnt) = struct('layer', {7, 5, 6}, ...
      'b_or_d', {'blob', 'diff', 'diff'}, 'w_or_r', {'response', w_or_r, w_or_r},
      'combine', combine);
end
cell_feat_opts

% combine fun
% input: actually a 2-dim matrix mxn
% output: a combined vector along dim 1 of size 1xn
% -----------------------------------------------------------------------------
function feat = l2(diff)
% -----------------------------------------------------------------------------
diff = squeeze(diff);
if size(diff, 2) ~= 1
  feat = sqrt(sum(diff.*diff, 1));
else
  feat = abs(diff)';
end
feat = normalize(feat);

% -----------------------------------------------------------------------------
function feat = l1(diff)
% -----------------------------------------------------------------------------
diff = abs(squeeze(diff));
if size(diff, 2) ~= 1
  feat = sum(diff, 1);
else 
  feat = diff';
end
feat = normalize(feat);

% -----------------------------------------------------------------------------
function feat = max_abs(diff)
% -----------------------------------------------------------------------------
diff = abs(squeeze(diff));
if size(diff, 2) ~= 1
  feat = max(diff, 1);
else
  feat = diff';
end
feat = normalize(feat);

% -----------------------------------------------------------------------------
function feat = max_pool(diff)
% -----------------------------------------------------------------------------
diff = squeeze(diff);
if size(diff, 2) ~= 1
  feat = max(diff, 1);
else 
  feat = diff';
end
feat = normalize(feat);

% -----------------------------------------------------------------------------
% input: a row vector
function feat = normalize(feat)
% -----------------------------------------------------------------------------
s = sum(feat);
if s ~= 0
  feat = feat/s;
end
