function draw_class_scatter(data, class, id, dir, name, num_class)
if ~exist('num_class', 'var')
  num_class = length(unique(class));
end
cmap = jet(num_class);
cmap(id, :) = [0,0,0];
f = figure();
for i = 1:num_class
  IX = find(class == i);
  class_data = data(IX,:);
  if (size(class_data,1) == 0)
    fprintf('ERR: %d 0!', i);
  end
  if size(class_data, 2) == 1
    scatter(class_data, ones(size(class_data,1),1)*i, 'markeredgecolor', cmap(i,:));
  else
    assert(size(class_data, 2) == 2);
    scatter3(class_data(:, 1), class_data(:, 2), ...
        ones(size(class_data,1),1)*i, 'markeredgecolor', cmap(i,:));
  end
  hold on;
end
title_str = [name '_' int2str(id)];
title(title_str);
saveas(gcf, [dir title_str '.jpg']);
saveas(gcf, [dir title_str '.fig']);
close(f);
