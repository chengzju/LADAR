save_name='xmtc_task'

python ensemble.py \
--data_dir ./data/Amazon-670K \
--test_file_path ./model_saved/Amazon-670K/${save_name} \
--ensemble_file_name xmtc_0.0001_0.001_fz2_swa3_fp1_id{} \
--ps_metric \

