%-------------------------------------------------------
function [counts, centers] = draw_hist_surf(data, num_bin, dir, suf)
%-------------------------------------------------------
% each column has its own histogram
%-------------------------------------------------------
fprintf('drawing %s num_bin = %d\n', suf, num_bin);
[counts, centers] = hist(data, num_bin);
fprintf('counts and centers calculated\n');
f = figure();
num_dim = size(data,2);
[X, Y] = meshgrid(1:num_dim, centers) ;
fprintf('meshgrided');
surf(X,Y,counts); colormap(jet);
fprintf('surf drawn');
title(suf);
view(2);
fprintf('saving at %s%s.jpg\n', dir, suf);
saveas(gcf,[dir suf '.jpg']);
saveas(gcf,[dir suf '.fig']);
fprintf('saved\n');
close(f);
