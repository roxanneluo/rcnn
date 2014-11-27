function ip = create_input_parser()
ip = inputParser;
ip.addRequired('feat_name',   @isstr);
ip.addParamValue('max_num_per_class', 5500,@isscalar);
ip.addParamValue('do_normalize',      true,@isscalar);
ip.addParamValue('do_lda',      false,@isscalar);
