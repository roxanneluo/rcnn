% ------------------------------------------------------------------------
function draw_summary(class_id, class, mean_pos, stdd, feat_name, folder, suffix, color)
% ------------------------------------------------------------------------
if ~exist('suffix', 'var')
  suffix = '';
end
if ~exist('color', 'var')
  color = 'b';
end

title_str = sprintf('%s-%d %s-summary-%s', class, class_id, feat_name, suffix);
title(title_str); hold on
%bar(mean_pos); hold on
%errorbar(mean_pos, stdd, 'linestyle', 'none');
errorbar(mean_pos, stdd, color);
saveas(gcf, [folder, title_str, '.jpg']);
