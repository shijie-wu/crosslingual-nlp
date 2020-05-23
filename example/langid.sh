src="all"
tgt="all"
task="langid"

ep=10
bs=32
lr=1e-3

model=bert-base-multilingual-cased # or xlm-roberta-base or xlm-mlm-100-1280
feature=6

data_path=/bigdata/dataset/langid
save_path=/bigdata/checkpoints/crosslingual-nlp

python src/train.py \
    --task $task \
    --data_dir $data_path \
    --trn_langs $src \
    --val_langs $tgt \
    --tst_langs $tgt \
    --pretrain $model \
    --freeze_layer 12 --feature_layer $feature --projector meanpool \
    --batch_size $bs \
    --learning_rate $lr \
    --max_epochs $ep \
    --schedule reduceOnPlateau \
    --gpus 1 \
    --default_save_path $save_path/$task/$(echo $src|tr ' ' '-')/$model \
    --exp_name feature_$feature