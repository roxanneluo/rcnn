function cache_weight_features(chunk, layer, backward_type, varargin)
ip = inputParser;
ip.addRequired('chunk', chunk,  @isstr);
ip.addRequired('layer', layer,  @isstr);
ip.addRequired('backward_type', backward_type, @isstr);
ip.addParamValue('do_normalize',  true,   @isscalar);
ip.addParamValue('do_lda',        true,   @isscalar);
ip.addParamValue('max_num_per_class', 5500,   @isscalar);
ip.parse(chunk, layer, backward_type, varargin{:});
opts = ip.Results;

% -------------------- CONFIG --------------------
feat_opt = struct('layer', layer, 'd', true, ...
    'w', true, 'combine', @l2, 'combine_name', 'l2', 'suf', backward_type);
cache_name  = [feat_opts_to_string(feat_opt) opts_name(opts)];
%cache_name   = feat_opts_to_string(feat_opt);
%net_file     = './data/caffe_nets/finetune_voc_2007_trainval_iter_70k';
net_file     = './data/nizf/nizf_model';
crop_mode    = 'warp';
crop_padding = 16;

VOCdevkit = './datasets/VOCdevkit2007';
imdb_train = imdb_from_voc(VOCdevkit, 'train', '2007');
imdb_val   = imdb_from_voc(VOCdevkit, 'val', '2007');
imdb_test  = imdb_from_voc(VOCdevkit, 'test', '2007');
imdb_trainval = imdb_from_voc(VOCdevkit, 'trainval', '2007');

switch chunk
  case 'train'
    cache_features(imdb_train, feat_opt, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'backward_type', backward_type, ...
        'cache_name', cache_name, ...
        'opts', opts);
    link_up_trainval(cache_name, imdb_train, imdb_trainval);
  case 'val'
    cache_features(imdb_val, feat_opt, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'backward_type', backward_type, ...
        'cache_name', cache_name, ...
        'opts', opts);
    link_up_trainval(cache_name, imdb_val, imdb_trainval);
  case 'test_1'
    end_at = ceil(length(imdb_test.image_ids)/2);
    cache_features(imdb_test, feat_opt, ...
        'start', 1, 'end', end_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'backward_type', backward_type, ...
        'cache_name', cache_name, ...
        'opts', opts);
  case 'test_2'
    start_at = ceil(length(imdb_test.image_ids)/2)+1;
    cache_features(imdb_test, feat_opt, ...
        'start', start_at, ...
        'crop_mode', crop_mode, ...
        'crop_padding', crop_padding, ...
        'net_file', net_file, ...
        'backward_type', backward_type, ...
        'cache_name', cache_name, ...
        'opts', opts);
end

% ------------------------------------------------------------------------
function link_up_trainval(cache_name, imdb_split, imdb_trainval)
% ------------------------------------------------------------------------
cmd = {['mkdir -p ./feat_cache/' cache_name '/' imdb_trainval.name '; '], ...
    ['cd ./feat_cache/' cache_name '/' imdb_trainval.name '/; '], ...
    ['for i in `ls -1 ../' imdb_split.name '`; '], ... 
    ['do ln -s ../' imdb_split.name '/$i $i; '], ... 
    ['done;']};
cmd = [cmd{:}];
fprintf('running:\n%s\n', cmd);
system(cmd);
fprintf('done\n');
