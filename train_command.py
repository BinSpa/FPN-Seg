# urur
python train_val.py --dataset 'URUR' --save_dir '../urur_exp' --batch_size 4 --slide_inference
python train_val.py \
--dataset 'GID' \
--save_dir '../gid_exp' \
--batch_size 8 \
--slide_inference \
--num_workers 10

python train_val.py \
--dataset 'FBP' \
--save_dir '../FBP' \
--batch_size 16 \
--slide_inference \
--num_workers 10 \
--mGPUs False