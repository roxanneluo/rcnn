export CUDA_VISIBLE_DEVICES=2
#matlab -nodisplay -r "train_all(3,6);quit;" | tee 3-fc7+l6 
matlab -nodisplay -r "train_all(3,5);" | tee 3-fc7+l5
