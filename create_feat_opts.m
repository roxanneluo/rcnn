function cell_feat_opts = create_feat_opts(proc, layer)
% feat_opt=
%   layer:
%   d: true for diff / false for blob
%   w: true for 'weight'/ false for 'response'
%   combine: @func_handle
%   combine_name: 'name_of_the_func'

test_num_layer = 3;
cell_feat_opts = {};
w_or_rs = {false};
b_or_ds = {false};
combine = @l2;
combine_name = 'l2';
dim = 1;

for wr = 1:length(w_or_rs)
  w_or_r = w_or_rs{wr};
  if ~ w_or_r
    start_layer = 5;
  else
    start_layer = 6;
  end
  % pure grad x each layer
  switch(proc)
  case 0
    feat_opts = struct('layer', 7, 'd', false, ...
        'w', w_or_r, 'combine', combine, 'combine_name', combine_name);
    cell_feat_opts = cat(dim, cell_feat_opts, feat_opts);
  case 1
%    for i=start_layer:start_layer+test_num_layer-1
    feat_opts = struct('layer', layer, 'd', true, ...
        'w', false, 'combine', combine, 'combine_name', combine_name);
    cell_feat_opts = cat(dim, cell_feat_opts, feat_opts);
  case 2
    % grad+act x each layer
    feat_opts = struct('layer', layer, 'd', {false, true},...
        'w', {false, false}, 'combine', combine, 'combine_name', combine_name);
    cell_feat_opts = cat(dim, cell_feat_opts, feat_opts);
  case 3
    % fc7 + (grad/act x pool5/fc6)
    feat_opts = struct('layer', {7, layer}, 'd', {false, false},...
        'w', false, 'combine', combine, 'combine_name', combine_name);
    cell_feat_opts = cat(dim, cell_feat_opts, feat_opts);
 case 4
  feat_opts = struct('layer', {7, layer}, 'd', {false, true},...
      'w', false, 'combine', combine, 'combine_name', combine_name);
  cell_feat_opts = cat(dim, cell_feat_opts, feat_opts);
 case 5
  feat_opts = struct('layer', layer, 'd', true,...
      'w', true, 'combine', combine, 'combine_name', combine_name);
  cell_feat_opts = cat(dim, cell_feat_opts, feat_opts);
 case 6
  feat_opts = struct('layer', {7, layer}, 'd', {false, true},...
      'w', {false,true}, 'combine', combine, 'combine_name', combine_name);
  cell_feat_opts = cat(dim, cell_feat_opts, feat_opts);
 otherwise
   feat_opts = struct('layer', {7, start_layer, start_layer+1}, ...
      'd', {false, true, true}, 'w', {false, w_or_r, w_or_r},...
      'combine', combine, 'combine_name', combine_name); 
   cell_feat_opts = cat(dim, cell_feat_opts, feat_opts);
  end
end
cell_feat_opts

% combine fun
% input: actually a 3-dim [combine_along_dim, combine_across_dim, num]
% output: a combined vector along dim 2 of size [combine_across_dim, num]
% -----------------------------------------------------------------------------
function feat = l2(diff)
% -----------------------------------------------------------------------------
feat = sqrt(sum(diff.*diff, 1));

% -----------------------------------------------------------------------------
function feat = l1(diff)
% -----------------------------------------------------------------------------
feat = sum(abs(diff), 1);

% -----------------------------------------------------------------------------
function feat = max_abs(diff)
% -----------------------------------------------------------------------------
feat = max(abs(diff), 1);

% -----------------------------------------------------------------------------
function feat = max_pool(diff)
% -----------------------------------------------------------------------------
feat = max(diff, 1);

