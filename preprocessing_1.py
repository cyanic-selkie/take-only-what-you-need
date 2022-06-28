import json
import faulthandler
import traceback
from unicodedata import normalize

import classla
import simdjson
import cutters
from tqdm import tqdm

faulthandler.enable()

classla.download("hr")
nlp = classla.Pipeline("hr", processors="tokenize,pos,lemma", use_gpu=True)

def normalize_unicode(text):
    return normalize("NFKC", text)


#def split_by_sentence(text):
#    sentences = cutters.cut(text, "hr")
#    to_return = []
#    for sentence in sentences:
#        to_return.append(sentence.str)
#    return to_return


def lemmatize(text):
    tokens = []
    lemmas = []
    for i, sentence in enumerate(nlp(text).sentences):
        tokens.append([])
        lemmas.append([])
        for word in sentence.words:
            tokens[-1].append(word.text)
            lemmas[-1].append(word.lemma)
    return tokens, lemmas


with open('data/dataset.json', 'rb') as f:
    dataset = simdjson.loads(f.read())


sentences = []

for datapoint in tqdm(dataset.values()):
    title = normalize_unicode(datapoint["title"])
    text = normalize_unicode(datapoint["text"])
    labels = datapoint["labels"]
    document_id = datapoint["filename"][:-4]
    
    tokens, lemmas = lemmatize(title)
    sentences.append({
        "document_id": document_id,
        "idx": 0,
        "tokens": tokens[0],
        "lemmas": lemmas[0],
        "labels": labels,
    })
    
    try:
        tokens, lemmas = lemmatize(text)

        for i, (tokens, lemmas) in enumerate(zip(tokens, lemmas)):
            sentences.append({
                "document_id": document_id,
                "idx": i + 1,
                "tokens": tokens,
                "lemmas": lemmas,
                "labels": labels,
            })
    except:
        continue


with open('data/preprocessed_1.json', 'w') as f:
    simdjson.dump(sentences, f, ensure_ascii=False)

