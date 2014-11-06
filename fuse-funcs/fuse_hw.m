function feat = fuse_hw(feat, dims)
num = size(feat,1);
dims = [dims(1)*dims(2), dims(3), dims(4)];
feat = reshape(feat, [num, dims]);
feat = feat.*feat;
feat = sqrt(sum(feat, 2));
dim = numel(feat) / num;

