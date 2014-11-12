function [IX, colormap] = draw_class_sort(data, baseIX)
[~, IX] = sort(data, 2);
colorIX = get_colorIX(baseIX);

dim = size(IX,2);
colormap = jet(dim);
colormap = colormap(colorIX,:);
imshow(IX, colormap);

%------------------------------------------------------
function colorIX = get_colorIX(classIX)
%------------------------------------------------------
dim = size(classIX,2);
map = [classIX', [1:dim]'];
map = sortrows(map);
colorIX = map(:,2);

