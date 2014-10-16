export CUDA_VISIBLE_DEVICES=1
matlab -nodisplay -r "train_all(2, 5);quit;" | tee 2-5grad+act
#matlab -nodisplay -r "train_all(2, 6);quit;" | tee 2-6grad+act
#matlab -nodisplay -r "train_all(2, 7);" | tee 2-7grad+act
