function cell_feat_opts = create_feat_opts()
% feat_opt=
%   layer:
%   b_or_d: 'blob'/'diff'
%   w_or_r: 'weight'/'response'
%   combine: @func_handle
%   combine_name: 'name_of_the_func'

test_num_layer = 2;
num_feat_cell = 2*(test_num_layer*2+4);
cell_feat_opts = cell(num_feat_cell,1);
w_or_rs = {'response', 'weight'};
b_or_ds = {'diff', 'blob'};
combine = @l2;
combine_name = 'l2';

cnt = 0;
for wr = 1:2
  w_or_r = w_or_rs{wr};
  if str_cmp(w_or_r, 'response')
    start_layer = 5;
  else
    start_layer = 6;
  % pure grad x each layer
  for i=start_layer:start_layer+test_num_layer-1
    cnt = cnt+1;
    cell_feat_opts{cnt} = struct('layer', i, 'b_or_d', 'diff', ...
        'w_or_r', w_or_r, 'combine', combine, 'combine_name', combine_name);
  end
  % grad+act x each layer
  for i=start_layer:start_layer+test_num_layer-1
    cnt = cnt+1;
    cell_feat_opts{cnt} = struct('layer', i, 'b_or_d', {'blob', 'diff'},...
        'w_or_r', {'response', w_or_r}, 'combine', combine, 'combine_name', combine_name);
  end
  % fc7 + (grad/act x pool5/fc6)
  for layer=start_layer:start_layer+test_num_layer-2
    for bd = 1:2
      b_or_d = b_or_ds{bd};
      cnt = cnt+1;
      cell_feat_opts{cnt} = struct('layer', {7, layer}, 'b_or_d', {'blob', b_or_d},...
          'w_or_r', w_or_r, 'combine', combine, 'combine_name', combine_name);
    end
  end
  %cnt = cnt+1;
  %cell_feat_opts{cnt} = struct('layer', {7, 5, 6}, ...
   %   'b_or_d', {'blob', 'diff', 'diff'}, 'w_or_r', {'response', w_or_r, w_or_r},...
   %   'combine', combine);
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
