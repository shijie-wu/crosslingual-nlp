# Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT

Shijie Wu and Mark Dredze. [*Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT*](https://arxiv.org/abs/1904.09077). EMNLP. 2019.

## Prerequisite

```bash
# Assuming this repo is save at `/path/to/crosslingual-nlp`
export CODE_DIR=/path/to/crosslingual-nlp
export SCRIPT_DIR=example/surprising-mbert

# Set envrionment variable `ROOT_DIR`, all related files will be store in this directory.
export ROOT_DIR=/bigdata
export DATA_DIR=$ROOT_DIR/dataset
export CKPT_DIR=$ROOT_DIR/checkpoints/crosslingual-nlp
```

<details>

<summary>Download dataset</summary>

```bash
mkdir -p $DATA_DIR && cd $DATA_DIR

# download MLDoc under $DATA_DIR/mldoc
mkdir -p $DATA_DIR/mldoc && cd $DATA_DIR/mldoc

# download XNLI
mkdir -p $DATA_DIR/xnli && cd $DATA_DIR/xnli
wget https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip
unzip https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip
mv XNLI-1.0/* ./
wget https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip
unzip https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip
mv XNLI-MT-1.0/* ./

# download CoNLL NER (02/03) and Chinese NER (Levow,2006) under $DATA_DIR/ner-conll
mkdir -p $DATA_DIR/ner-conll && cd $DATA_DIR/ner-conll

# download Universal Dependency (v1.4 and v2.2)
mkdir -p $DATA_DIR/universaldependencies && cd $DATA_DIR/universaldependencies
# download https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1827
# and unzip ud-treebanks-v1.4.tgz under $DATA_DIR/universaldependencies

# download https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2837
# and unzip ud-treebanks-v2.2.tgz under $DATA_DIR/universaldependencies
# please follow Appendix 1 of https://arxiv.org/abs/1811.00570 to concat different
# treebanks of the same language into one treebank with language code as foler name

# download language id dataset
mkdir -p $DATA_DIR/langid && cd $DATA_DIR/langid
# download https://martin-thoma.com/wili/ and process the dataset under $DATA_DIR/langid
# or feel free to contact shijie dot wu at jhu dot edu to get a processed version
```

</details>


## Evaluation

```bash
cd $CODE_DIR
export ENCODER=bert-base-multilingual-cased
```

### Evaluate mBERT on 5 tasks
```bash
seed=42
freeze=-1 # freezing bottom X layers including embeddings layer, -1 => finetune all layers

# evalute pretrained encoder
for task in mldoc xnli ner-conll udpos parsing; do
bash $SCRIPT_DIR/evaluate.sh $seed $ENCODER $task $freeze
done

# check results
for task in mldoc xnli ner-conll udpos parsing; do
python scripts/compile_result.py best $CKPT_DIR/$task/0-shot-finetune-freeze$freeze/$ENCODER
done
```

### Probe mBERT on language ID
```bash
seed=42
feature=6 # take feature from layer X-th (including embeddings layer as 0-th layer)

# probe pretrained encoder on language id
bash $SCRIPT_DIR/langid.sh $seed $ENCODER $feature
# check results
python scripts/compile_result.py best $CKPT_DIR/langid/$ENCODER/feature_$feature
```
