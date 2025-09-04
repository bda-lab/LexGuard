# term_extractor_infer.py
from transformers import BigBirdTokenizerFast, BigBirdForTokenClassification
import torch

tokenizer = BigBirdTokenizerFast.from_pretrained("./term_extractor_out/model")
model = BigBirdForTokenClassification.from_pretrained("./term_extractor_out/model")
label_map = {0:"O",1:"B-TERM",2:"I-TERM",3:"B-DEF",4:"I-DEF"}

def extract_terms(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    with torch.no_grad():
        logits = model(**inputs).logits
        preds = logits.argmax(-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    labels = [label_map[p] for p in preds]
    # merge tokens to phrases -> return term-definition pairs (postprocess)
    return tokens, labels

if __name__ == "__main__":
    txt = open("data/regulation_sample.txt").read()
    print(extract_terms(txt))
