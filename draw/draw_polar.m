function draw_polar(id, angle, data_norm, dir, title_str)
fprintf('drawing polar plot of %s\n', title_str);
cmap = jet(size(angle,2));
cmap(id,:) = [0,0,0];
f = figure();
polar(angle(:,1), data_norm, '.');
hold on;
title(title_str);
for i = 1:size(angle,2)
  ph = polar(angle(:,i), data_norm, '.'); hold on;
  set(ph, 'markeredgecolor', cmap(i,:));
end
saveas(gcf, [dir, title_str, '.jpg']);
saveas(gcf, [dir, title_str, '.fig']);
close(f);
