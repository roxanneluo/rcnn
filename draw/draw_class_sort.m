function [IX, colormap] = draw_class_sort(IX, baseIX)
colorIX = get_colorIX(baseIX);

dim = size(IX,2);
colormap = jet(dim);
colormap = colormap(colorIX,:);

%------------------------------------------------------
function colorIX = get_colorIX(classIX)
%------------------------------------------------------
dim = size(classIX,2);
map = [classIX', [1:dim]'];
map = sortrows(map);
colorIX = map(:,2);

