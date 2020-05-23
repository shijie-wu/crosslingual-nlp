src="en" # training languages, could be more than 1
tgt="de en es zh" # dev & test languages, also used for model selection (could be decoupled)
task="mldoc" # change it to "xnli" to run xnli instead

ep=3
bs=32
lr=5e-5

model=bert-base-multilingual-cased # or xlm-roberta-base or xlm-mlm-100-1280

data_path=/bigdata/dataset/$task
save_path=/bigdata/checkpoints/crosslingual-nlp

# train 10k steps, wramup 1k steps, validate every 200 steps
# --max_steps 10000 --warmup_steps 1000 --val_check_interval 200

python src/train.py \
    --task $task \
    --data_dir $data_path \
    --trn_langs $src \
    --val_langs $tgt \
    --tst_langs $tgt \
    --pretrain $model \
    --batch_size $bs \
    --learning_rate $lr \
    --max_epochs $ep \
    --warmup_portion 0.1 \
    --gpus 1 \
    --default_save_path $save_path/$task/$(echo $src|tr ' ' '-')/$model \
    --exp_name bs$bs-lr$lr-ep$ep
