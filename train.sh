CUDA_VISIBLE_DEVICES=2 python train_main.py --root_path="data/HAM10000" \
--csv_file_train="data/skin_split/train.csv" \
--csv_file_val="data/skin_split/validation.csv" \
--csv_file_test="data/skin_split/test.csv" \
--model_path="models/testing/skin.pth" \
--warmup=5 \
--gpu=2 \
--mu=0.1 \
--dataset="skin"\