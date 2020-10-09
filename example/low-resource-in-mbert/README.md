# Are All Languages Created Equal in Multilingual BERT?

Shijie Wu and Mark Dredze. [*Are All Languages Created Equal in Multilingual BERT?*](https://arxiv.org/abs/2005.09093) RepL4NLP. 2020.

## Prerequisite

```bash
# Assuming this repo is save at `/path/to/crosslingual-nlp`
export CODE_DIR=/path/to/crosslingual-nlp
export SCRIPT_DIR=example/low-resource-in-mbert

# Set envrionment variable `ROOT_DIR`, all related files will be store in this directory.
export ROOT_DIR=/bigdata
export DATA_DIR=$ROOT_DIR/dataset
export CKPT_DIR=$ROOT_DIR/checkpoints/crosslingual-nlp
```

<details>

<summary>Download dataset</summary>

```bash
mkdir -p $DATA_DIR && cd $DATA_DIR

# download WikiANN
mkdir -p $DATA_DIR/ner-wiki && cd $DATA_DIR/ner-wiki
# download https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN
# and put each language folder under $DATA_DIR/ner-wiki

# download Universal Dependency (v2.3)
mkdir -p $DATA_DIR/universaldependencies && cd $DATA_DIR/universaldependencies
# download https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2895
# and unzip ud-treebanks-v2.3.tgz under $DATA_DIR/universaldependencies
```

</details>


## Evaluation

```bash
cd $CODE_DIR
export ENCODER=bert-base-multilingual-cased
```

### Evaluate mBERT on WikiANN
```bash
lang=en # take English as an example
seed=42

# evalute pretrained encoder
bash $SCRIPT_DIR/evaluate.sh $seed $ENCODER ner-wiki $lang
# check results
python scripts/compile_result.py best $CKPT_DIR/ner-wiki/monolingual/$lang/$ENCODER
```

### Evaluate mBERT on UD POS tagging and UD dependency parsing
```bash
lang=English-EWT # take English as an example
seed=42

# evalute pretrained encoder
for task in udpos parsing; do
bash $SCRIPT_DIR/evaluate.sh $seed $ENCODER $task $lang
done

# check results
for task in udpos parsing; do
python scripts/compile_result.py best $CKPT_DIR/$task/monolingual/$ENCODER
done
```
