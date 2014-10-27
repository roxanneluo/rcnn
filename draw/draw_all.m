% ------------------------------------------------------------------------
function draw_all(class_id, class, X_pos, feat_name, folder, suffix)
% ------------------------------------------------------------------------
if ~exist('suffix', 'var')
  suffix = '';
end
figure();
title_str = sprintf('%s-%d %s-all-%s', class, class_id, feat_name, suffix);
title(title_str); hold on;
fprintf('before surf\n');
surf(X_pos); hold on;
fprintf('after surf\n');
xlabel('num');
ylabel('feature dim');
fprintf('after label\n')
[folder, title_str, '.jpg']
saveas(gcf, [folder, title_str, '.jpg']);
saveas(gcf, [folder, title_str, '.fig']);
fprintf('after saveas');

