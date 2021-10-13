# Do Explicit Alignments Robustly Improve Multilingual Encoders?

Mahsa Yarmohammadi*, Shijie Wu*, Marc Marone, Haoran Xu, Seth Ebner, Guanghui Qin, Yunmo Chen, Jialiang Guo, Craig Harman, Kenton Murray, Aaron Steven White, Mark Dredze, and Benjamin Van Durme. [*Everything Is All It Takes: A Multipronged Strategy for Zero-Shot Cross-Lingual Information Extraction*](https://arxiv.org/abs/2109.06798) EMNLP. 2021.

## Prerequisite

```bash
export SCRIPT_DIR=example/data-projection

# Set envrionment variable `ROOT_DIR`, all related files will be store in this directory.
export ROOT_DIR=/bigdata
export DATA_DIR=$ROOT_DIR/dataset
export CKPT_DIR=$ROOT_DIR/checkpoints/clnlp
```

<details>

<summary>Download dataset</summary>

```bash
mkdir -p $DATA_DIR && cd $DATA_DIR

# download WikiANN
mkdir -p $DATA_DIR/ner-wiki && cd $DATA_DIR/ner-wiki
# download https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN
# and put each language folder under $DATA_DIR/ner-wiki

# download Universal Dependency (v2.7)
mkdir -p $DATA_DIR/universaldependencies && cd $DATA_DIR/universaldependencies
# download https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3424
# and unzip ud-treebanks-v2.7.tgz under $DATA_DIR/universaldependencies
```

</details>

## Data Projection

To create silver data for any target language, we first translate the text into target language, then either align and project the supervision (data projection) or label the text using zero-shot model (self-training).

```bash
# source langauge
export SRC=en
# target langauge
export TGT=ar
# we use a publicly available MT system as an example
export MT_SYSTEM=Helsinki-NLP/opus-mt-$SRC-$TGT

# alignment encoder (no fine-tuning)
# for Arabic as a target language, we could use the following combination:
# export ALIGN_ENCODER=jhu-clsp/roberta-large-eng-ara-128k    # or lanwuwei/GigaBERT-v4-Arabic-and-English
# export ALIGN_LAYER=16                                       # or 8
# export ALIGN_NAME=L128K_l16                                 # or giga4_l8
export ALIGN_ENCODER=xlm-roberta-large # or bert-base-multilingual-cased
export ALIGN_LAYER=16                  # or 8
export ALIGN_NAME=xlmrl_l16            # or mbert_l8

# downstream encoder
# for Arabic as a target language, we could use the following combination:
# export TASK_ENCODER=jhu-clsp/roberta-large-eng-ara-128k    # or lanwuwei/GigaBERT-v4-Arabic-and-English
export TASK_ENCODER=xlm-roberta-large # or bert-base-multilingual-cased
export TASK_ENCODER_NAME=$(echo "$TASK_ENCODER"|cut -d/ -f2)
```
### Data Projection
```bash
# Parsing + POS (UD 2.7)
for split in train dev test; do
  bash $SCRIPT_DIR/project-ud.sh $split $TGT $ALIGN_ENCODER $ALIGN_LAYER $ALIGN_NAME
done

# NER (WikiAnn)
for split in train dev test; do
  bash $SCRIPT_DIR/project-ner.sh $split $TGT $ALIGN_ENCODER $ALIGN_LAYER $ALIGN_NAME
done
```
### Self-training
```bash
# run data projection above first
#
# Parsing + POS (UD 2.7)
# train a zero-shot model
seed=1111
bash $SCRIPT_DIR/evaluate-0shot.sh $seed $TASK_ENCODER udpos
bash $SCRIPT_DIR/evaluate-0shot.sh $seed $TASK_ENCODER parsing
# label the data using the zero-shot model
for split in train dev test; do
  bash $SCRIPT_DIR/self-training.sh ud $split $TGT $TASK_ENCODER
done

# NER (WikiAnn)
# train a zero-shot model
bash $SCRIPT_DIR/evaluate-0shot.sh $seed $TASK_ENCODER ner-wiki
# label the data using the zero-shot model
for split in train dev test; do
  bash $SCRIPT_DIR/self-training.sh ner $split $TGT $TASK_ENCODER
done
```

## Evaluation

### Baseline (Zero-shot)
```bash
for seed in 1111 3333 5555; do
  for task in ner-wiki udpos parsing; do
    bash $SCRIPT_DIR/evaluate-0shot.sh $seed $TASK_ENCODER $task
  done
done

# check results
for task in ner-wiki udpos parsing; do
  python scripts/compile_result.py mean_std $CKPT_DIR/en.0shot/$task/$TASK_ENCODER_NAME
done
```

### Evaluate Data Projection
```bash
system="helsinki_opus.$ALIGN_NAME"
for seed in 1111 3333 5555; do
  for task in ner-wiki udpos parsing; do
    bash $SCRIPT_DIR/evaluate-silver.sh $seed $TASK_ENCODER $task $system $TGT
  done
done

# check results
for task in ner-wiki udpos parsing; do
  python scripts/compile_result.py mean_std $CKPT_DIR/en-$TGT.silver.$system/$task/$TASK_ENCODER_NAME
done
```

### Evaluate Self-training
```bash
case "$TASK_ENCODER" in
"bert-base-multilingual-cased")             system="helsinki_opus.self_mbert" ;;
"xlm-roberta-large")                        system="helsinki_opus.self_xlmrl" ;;
"lanwuwei/GigaBERT-v4-Arabic-and-English")  system="helsinki_opus.self_giga4" ;;
"jhu-clsp/roberta-large-eng-ara-128k")      system="helsinki_opus.self_L128K" ;;
*) echo ERROR; exit ;;
esac

for seed in 1111 3333 5555; do
  for task in ner-wiki udpos parsing; do
    bash $SCRIPT_DIR/evaluate-silver.sh $seed $TASK_ENCODER $task $system $TGT
  done
done

# check results
for task in ner-wiki udpos parsing; do
  python scripts/compile_result.py mean_std $CKPT_DIR/en-$TGT.silver.$system/$task/$TASK_ENCODER_NAME
done
```
