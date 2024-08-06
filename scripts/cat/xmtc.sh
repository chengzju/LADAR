save_name='xmtc_task'
id=1

python -m torch.distributed.launch --nproc_per_node=8 --master_port='1308' --master_addr='127.0.0.1' xmtc_main_ddp.py \
--data_dir ./data/AmazonCat-13K \
--maxlength 400 \
--epochs 40 \
--batch_size 32 \
--lr 1e-3 \
--encoder_lr 1e-4 \
--swa_warmup 2 \
--save_path ./model_saved/AmazonCat-13K/${save_name} \
--train_model \
--droprate 0.3 \
--feature_mode cls \
--gpuid '0,1,2,3,4,5,6,7' \
--last_freeze_layer -1 \
--try_num ${id} \
--seed 123 \
--sample_layer 5 \
--atten_num 50 \
--num_frag 40 \
--len_frag 10 \
--sample_test \
--sample_test_size 5000 \
--fp16 \
--ps_metric \

python -m torch.distributed.launch --nproc_per_node=1 --master_port='1311' --master_addr='127.0.0.1' xmtc_main_ddp.py \
--data_dir ./data/AmazonCat-13K \
--maxlength 400 \
--save_path ./model_saved/AmazonCat-13K/${save_name} \
--test_model \
--batch_size 48 \
--test_file_path xmtc_0.0001_0.001_fz-1_swa2_fp1_id${id} \
--droprate 0.3 \
--feature_mode cls \
--gpuid '0' \
--sample_layer 5 \
--atten_num 50 \
--num_frag 40 \
--len_frag 10 \
--fp16 \
--ps_metric \
--test_epoch 20 \

