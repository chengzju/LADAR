save_name='xmtc_task'

python ensemble.py \
--data_dir ./data/AmazonCat-13K \
--test_file_path ./model_saved/AmazonCat-13K/${save_name} \
--ensemble_file_name xmtc_0.0001_0.001_fz-1_swa2_fp1_id{} \
--ps_metric \
