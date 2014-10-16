export CUDA_VISIBLE_DEVICES=3
matlab -nodisplay -r "train_all(4);" | tee 4-comb-all
