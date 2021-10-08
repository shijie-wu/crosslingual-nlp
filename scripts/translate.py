import fire
import jieba
from sacremoses import MosesTokenizer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

LANG_MAPPING = {"ar": "ara", "vi": "vie", "zh": "cmn_Hant"}
NO_PREFIX_LANGS = set(["de", "es", "fr", "hi", "ru"])


def main(
    infile,
    model_name="Helsinki-NLP/opus-mt-en-ar",
    src="en",
    tgt="ar",
    batch_size=32,
    num_beams=4,
    length_penalty=0.6,
    no_repeat_ngram_size=3,
    original=False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.eval()
    model = model.to("cuda")

    mt = MosesTokenizer(lang=tgt)

    if "m2m100" in model_name:
        tokenizer.src_lang = src

    data = []
    with open(infile) as fp:
        for line in fp.readlines():
            text = line.strip()
            if "opus-mt" in model_name:
                if tgt not in NO_PREFIX_LANGS:
                    assert tgt in LANG_MAPPING
                    text = f">>>{LANG_MAPPING[tgt]}<<< " + text
            data.append(text)

    for idx in tqdm(range(len(data))[::batch_size]):
        batch = tokenizer(
            data[idx : idx + batch_size],
            return_tensors="pt",
            padding=True,
            max_length=tokenizer.max_len_single_sentence,
            truncation=True,
        )
        batch = batch.to("cuda")
        if "m2m100" in model_name:
            translated = model.generate(
                **batch,
                forced_bos_token_id=tokenizer.get_lang_id(tgt),
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        else:  # "Helsinki-NLP" in model_name
            translated = model.generate(
                **batch,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        for t in translated:
            text = tokenizer.decode(t, skip_special_tokens=True)
            if not original:
                if tgt == "zh":
                    text = " ".join(jieba.cut(text))
                    text = " ".join(text.split())
                else:
                    text = mt.tokenize(text, escape=False)
                    text = " ".join(text)
            print(text)


if __name__ == "__main__":
    fire.Fire(main)
