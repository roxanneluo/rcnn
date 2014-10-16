export CUDA_VISIBLE_DEVICES=1
matlab -nodisplay -r "addpath fuse-funcs/; train_all(0, 7);" | tee 0-pure-act
