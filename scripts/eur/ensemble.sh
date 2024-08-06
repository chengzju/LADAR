save_name='xmtc_task'

python ensemble.py \
--data_dir ./data/EUR-Lex \
--test_file_path ./model_saved/EUR-Lex/${save_name} \
--ensemble_file_name xmtc_0.0001_0.001_fz6_swa5_fp0_id{} \
--ps_metric \

