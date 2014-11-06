function prob = softmax(feat)
feat = feat - repmat(max(feat, [], 2), [1, size(feat, 2)]);
exponent = exp(feat);
s = sum(exponent, 2);
prob = exponent ./ repmat(s, [1, size(feat,2)]);
