function draw_all_sort(data, nums, pre, dir)
[~,IX] = sort(data, 2);
dim = size(IX,2);
%globalIX = get_baseIX(IX);
globalIX = get_baseIX(data);

VOCdevkit = './datasets/VOCdevkit2007';
imdb = imdb_from_voc(VOCdevkit, 'trainval', '2007');
num_class = length(imdb.class_ids)+1;
classes = {imdb.classes{:}, 'background'};

classIX = zeros(num_class, size(IX,2));
num_start = 1;
for i = 1:num_class
data, nums, pre  num_end = num_start + nums(i)-1;
  %classIX(i,:) = get_baseIX(IX(num_start:num_end,:));
  classIX(i,:) = get_baseIX(data(num_start:num_end,:));
  draw_class_sort(dir, i, classes{i}, IX(num_start:num_end,:), classIX(i,:), [pre 'data_class']);
  draw_class_sort(dir, i, classes{i}, IX(num_start:num_end,:), globalIX, [pre 'data_global']);
  num_start = num_end+1;
end

draw_class_sort(dir, 0, 'IX', classIX, globalIX, pre );

%------------------------------------------------------
function draw_class_sort(dir, class_id, class, IX, baseIX, prefix)
%------------------------------------------------------
colorIX = get_colorIX(baseIX);

dim = size(IX,2);
im_file = sprintf('%s%sSort-%d-%s.jpg', dir, prefix, class_id, class);
colormap = jet(dim);
colormap = colormap(colorIX,:);
size(colormap)
imwrite(IX, colormap, im_file);

%------------------------------------------------------
function colorIX = get_colorIX(classIX)
%------------------------------------------------------
dim = size(classIX,2);
map = [classIX', [1:dim]'];
map = sortrows(map);
colorIX = map(:,2);


%{
%------------------------------------------------------
function classIX = get_baseIX(IX)
%------------------------------------------------------
dim = size(IX,2);
cnt = zeros(size(IX));
for i = 1:size(IX,1)
  cnt(i,IX(i,:)) = 1:dim;
end
dim_cnt = sum(cnt);
[~, classIX] = sort(dim_cnt)
%}
