# Do Explicit Alignments Robustly Improve Multilingual Encoders?

Shijie Wu and Mark Dredze. [*Do Explicit Alignments Robustly Improve Multilingual Encoders?*](https://arxiv.org/abs/2010.02537) EMNLP. 2020.

## Prerequisite

```bash
# Assuming this repo is save at `/path/to/crosslingual-nlp`
export CODE_DIR=/path/to/crosslingual-nlp
export SCRIPT_DIR=example/contrastive-alignment

# Set envrionment variable `ROOT_DIR`, all related files will be store in this directory.
export ROOT_DIR=/bigdata
export DATA_DIR=$ROOT_DIR/dataset
export CKPT_DIR=$ROOT_DIR/checkpoints/crosslingual-nlp
```

<details>

<summary>Set up environment for bitext preparation</summary>

```bash
mkdir -p $ROOT_DIR/dataset/bitext

# Install fast_align (https://github.com/clab/fast_align)
cd $ROOT_DIR/dataset/bitext
git clone git@github.com:clab/fast_align.git
mkdir fast_align/build && cd fast_align/build
cmake .. && make

# Install XLM (https://github.com/facebookresearch/XLM)
cd $ROOT_DIR/dataset/bitext
git clone git@github.com:facebookresearch/XLM.git
cd XLM
./install-tools.sh

# Download bitext
lg_pairs="ar-en de-en en-es en-fr en-hi en-ru en-vi en-zh"
for lg_pair in $lg_pairs; do
    ./get-data-para.sh $lg_pair
done

# Link tokenizer
ln -s $ROOT_DIR/dataset/bitext/XLM/tools $CODE_DIR/tools
```
</details>

```bash
cd $CODE_DIR
```

Prepare bitext used by [XLM](https://github.com/facebookresearch/XLM) and sample 1M subset for each languages
```bash
for lng in ar de es fr hi ru vi zh; do
    bash $SCRIPT_DIR/tokenize-xlm.sh $lng
done
```
Prepare bitext from [OPUS-100](https://github.com/EdinburghNLP/opus-100-corpus)
```bash
for lng in ar de es fr hi ru vi zh; do
    bash $SCRIPT_DIR/tokenize-opus.sh $lng
done
```

## Alignment

Align multilingual encoder with bitext
```bash
cd $CODE_DIR
export ENCODER=xlm-roberta-base # or bert-base-multilingual-cased
export BITEXT=xlm               # or opus
```
### Linear Alignment
```bash
for LANG_PAIR in en-ar en-de en-es en-fr en-hi en-ru en-vi en-zh; do
bash $SCRIPT_DIR/alignment-linear.sh $ENCODER $LANG_PAIR $BITEXT
done
```
### L2 Alignment ([Cao et al. (2020)](https://openreview.net/forum?id=r1xCMyBtPS))
```bash
bash $SCRIPT_DIR/alignment-l2.sh $ENCODER $BITEXT
```
### Weak Contrastive Alignment (OUR)
```bash
bash $SCRIPT_DIR/alignment-cntrstv.sh cntrstv $ENCODER $BITEXT
```
### Strong Contrastive Alignment (OUR)
```bash
bash $SCRIPT_DIR/alignment-cntrstv.sh jnt_cntrstv $ENCODER $BITEXT
```

## Evaluation

<details>

<summary>Download evaluation dataset</summary>

```bash
mkdir -p $DATA_DIR && cd $DATA_DIR

# download XNLI
mkdir -p $DATA_DIR/xnli && cd $DATA_DIR/xnli
wget https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip
unzip https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip
mv XNLI-1.0/* ./
wget https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip
unzip https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip
mv XNLI-MT-1.0/* ./

# download WikiANN
mkdir -p $DATA_DIR/ner-wiki && cd $DATA_DIR/ner-wiki
# download https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN
# and put each language folder under $DATA_DIR/ner-wiki

# download Universal Dependency (v2.6)
mkdir -p $DATA_DIR/universaldependencies && cd $DATA_DIR/universaldependencies
# download https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3226
# and unzip ud-treebanks-v2.6.tgz under $DATA_DIR/universaldependencies
```

</details>

```bash
cd $CODE_DIR
```

### Baseline
```bash
# evalute pretrained encoder
bash $SCRIPT_DIR/evaluate-all.sh finetune $ENCODER

# check results
for task in xnli ner-wiki udpos parsing; do
python scripts/compile_result.py mean_std $CKPT_DIR/$task/0-shot-finetune/$ENCODER
done
```

### Evaluate linear mapping
```bash
# train feature-based English model
bash $SCRIPT_DIR/evaluate-all.sh feature $ENCODER
# collect linear mapping
python $SCRIPT_DIR/dump.py linear --root_dir $ROOT_DIR --data $BITEXT --model $ENCODER

# evaluate linear mapping
seed=42 # dummy seed (no randomness at evaluation)
mapping=linear-orth0.01
for task in xnli ner-wiki udpos parsing; do
bash $SCRIPT_DIR/evaluate-mapping.sh $seed $ENCODER $task feature $mapping
done

# check results
for task in xnli ner-wiki udpos parsing; do
python scripts/compile_result.py mean_std $CKPT_DIR/$task/0-shot-feature-map-$mapping/$ENCODER
done
```

### Evaluate aligned encoder (L2 Alignment or Weak/Strong Contrastive Alignment)
```bash
# convert checkpoint to a folder accepted by `transformers.AutoModel.from_pretrained`
python $SCRIPT_DIR/dump.py single /path/to/checkpoint/final.ckpt /path/to/pretrained/aligned_encoder
# evalute aligned encoder
bash $SCRIPT_DIR/evaluate-all.sh finetune /path/to/pretrained/aligned_encoder

# check results
for task in xnli ner-wiki udpos parsing; do
python scripts/compile_result.py mean_std $CKPT_DIR/$task/0-shot-finetune/aligned_encoder
done
```
