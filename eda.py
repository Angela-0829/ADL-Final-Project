from textattack.augmentation import WordNetAugmenter
from random_word import RandomWords
import random
import numpy as np
wordnet_aug = WordNetAugmenter()

def EDA(text, option, count):
    if option == "replace":
        for _ in range(count):
            text = wordnet_aug.augment(text)[0]
        return text
    if option == "delete":
        text_s = text.split()
        for _ in range(count):
            text_s.pop(random.randint(0, len(text_s)-1))
            if len(text_s) == 1:
                break
        return " ".join(text_s)
    if option == "swap":
        text_s = text.split()
        index_list = list(np.arange(len(text_s)))
        for _ in range(count):
            num = random.sample(index_list, 2)
            text_s[num[0]], text_s[num[1]] = text_s[num[1]], text_s[num[0]]
        return " ".join(text_s)
    if option == "insert":
        rw = RandomWords()
        text_s = text.split()
        for _ in range(count):
            words = rw.get_random_word()
            text_s.insert(random.randint(0, len(text_s)), words)
        return " ".join(text_s)
