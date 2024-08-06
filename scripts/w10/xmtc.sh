save_name='xmtc_task'

python xmtc_main.py \
--data_dir ./data/Wiki10-31K \
--maxlength 500 \
--epochs 80 \
--batch_size 16 \
--encoder_lr 1e-4 \
--lr 1e-3 \
--save_path ./model_saved/Wiki10-31K/${save_name} \
--train_model \
--droprate 0.7 \
--feature_mode avg \
--last_freeze_layer 6 \
--try_num 1 \
--seed 234 \
--sample_layer 5 \
--atten_num 30 \
--num_frag 5 \
--len_frag 100 \
--test_model \
