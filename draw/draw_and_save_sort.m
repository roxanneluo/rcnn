function draw_and_save_sort(data, nums, dir, title, commonIX)
if exist('IX', 'var')
  commonIX = get_base_IX(data);
end
IX = sort(data, 2);
[IX, colormap] = draw(dir, IX, commonIX);
f = figure();
imshow(IX, colormap);
saveas(gcf, [dir title '.jpg']);
close(f);
