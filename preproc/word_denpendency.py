import os
import json
from tqdm import tqdm
from nltk.corpus import wordnet
from collections import defaultdict

dataset = 'MSRVTT-QA'
DIR = f'data/{dataset}'


if __name__ == '__main__':
    with open(os.path.join(DIR, 'vocab.json')) as f:
        vocab = json.load(f)

    words = list(vocab.keys())
    vocab_tree = defaultdict(list)
    count = 0
    t = tqdm(words)

    for word1 in t:
        concept1 = wordnet.synsets(word1)
        if concept1:
            concept1_path = set(h for p in concept1[0].hypernym_paths() for h in p)
            for word2 in words:
                if word1 != word2:
                    concept2 = wordnet.synsets(word2)
                    if concept2 and concept2[0] in concept1_path:
                        vocab_tree[word1].append(word2)
        else:
            count += 1
        t.set_postfix(ignore=count)

    with open(os.path.join(DIR, 'vocab_tree.json'), 'w') as f:
        json.dump(vocab_tree, f, indent=4)
