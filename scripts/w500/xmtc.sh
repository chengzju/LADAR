save_name='xmtc_task'
id=1

python -m torch.distributed.launch --nproc_per_node=8 --master_port='1316' --master_addr='127.0.0.1' xmtc_main_plt_ddp.py \
--data_dir ./data/Wiki-500K \
--maxlength 300 \
--epochs 40 \
--batch_size 32 \
--lr 1e-3 \
--encoder_lr 1e-4 \
--swa_warmup 3 \
--save_path ./model_saved/Wiki-500K/${save_name} \
--train_model \
--droprate 0.5 \
--feature_mode cls \
--gpuid '0,1,2,3,4,5,6,7' \
--last_freeze_layer 2 \
--try_num ${id} \
--seed 234 \
--sample_layer 5 \
--atten_num 16 \
--num_frag 3 \
--len_frag 100 \
--fp16 \
--candidates_num 2000 \
--candidates_topk 32 \
--sample_test \
--sample_test_size 5000 \
--hidden_size 300 \
--cluster_id 0 \
--ps_metric \

python -m torch.distributed.launch --nproc_per_node=1 --master_port='1646' --master_addr='127.0.0.1' xmtc_main_plt_ddp.py \
--data_dir ./data/Wiki-500K \
--maxlength 300 \
--batch_size 48 \
--save_path ./model_saved/Wiki-500K/${save_name} \
--test_model \
--test_file_path xmtc_0.0001_0.001_fz2_swa3_fp1_id${id} \
--droprate 0.5 \
--feature_mode cls \
--gpuid '0' \
--sample_layer 5 \
--atten_num 16 \
--patch_num 3 \
--patch_len 100 \
--fp16 \
--candidates_num 2000 \
--candidates_topk 32 \
--hidden_size 300 \
--cluster_id 0 \
--test_epoch 20 \
--ps_metric \
