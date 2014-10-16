export CUDA_VISIBLE_DEVICES=0
matlab -nodisplay -r "train_all(1, 5);quit();"
matlab -nodisplay -r "train_all(1, 6);"
