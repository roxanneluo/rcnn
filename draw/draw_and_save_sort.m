function draw_and_save_sort(data, dir, title, commonIX)
if exist('IX', 'var')
  commonIX = get_base_IX(data);
end
f = figure();
draw_class_sort(data, commonIX);
saveas(gcf, [dir title '.jpg']);
close(f);
