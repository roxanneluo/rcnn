function train_pca(part_id)
pca_ratio = 0.6:0.1:0.9;
if part_id == 1
  pca_ratio = pca_ratio(1:2);
else 
  pca_ratio = pca_ratio(3:end);
end

for i = 1:length(pca_ratio)
  train_all(6,5,true,true,true, false, pca_ratio(i));
end
