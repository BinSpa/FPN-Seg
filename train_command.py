# urur
python train_val.py --dataset 'URUR' --save_dir '../urur_exp' --batch_size 4 --slide_inference
python train_val.py \
--dataset 'GID' \
--save_dir '../GID' \
--batch_size 16 \
--slide_inference \
--num_workers 10 \
--mGPUs False \
--epochs 21000 \
--eval_interval 2100

python train_val.py \
--dataset 'FBP' \
--save_dir '../FBP' \
--batch_size 16 \
--slide_inference \
--num_workers 10 \
--mGPUs False \
--epochs 21000 \
--eval_interval 2100