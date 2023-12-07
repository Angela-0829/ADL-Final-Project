'''This file is used to evaluate the output sentences of the model'''
import string
import nltk
import evaluate


def remove_eos(pred: list[str]):
    '''Remove end of sentence token from the output of the model'''
    for i, sen in enumerate(pred):
        pred[i] = sen.replace('<|endoftext|>', '')


def get_rouge(pred: list[str], target: list[str]):
    '''Calculate the rouge score of the model output'''
    rouge = evaluate.load('rouge')
    return rouge.compute(predictions=pred, references=target)


def get_bleu(pred: list[str], target: list[str]):
    '''Calculate the bleu score of the model output'''
    cands_list_bleu = [sentence.split() for sentence in pred]
    refs_list_bleu = [[sentence.split()] for sentence in target]
    bleu_score = nltk.translate.bleu_score.corpus_bleu(
        refs_list_bleu, cands_list_bleu)
    bleu_score_1 = nltk.translate.bleu_score.corpus_bleu(
        refs_list_bleu, cands_list_bleu, weights=(1, 0, 0, 0))
    bleu_score_2 = nltk.translate.bleu_score.corpus_bleu(
        refs_list_bleu, cands_list_bleu, weights=(0.5, 0.5, 0, 0))

    return {'bleu1': bleu_score_1, 'bleu2': bleu_score_2, 'bleu': bleu_score}


def punctuation_remove(sent_list: list[str]):
    '''Remove punctuation from the model output'''
    # create a translation table that maps punctuation to None
    table = str.maketrans('', '', string.punctuation)
    # apply the translation table to each string in the list
    return [s.translate(table) for s in sent_list]


def exact_match(pred: list[str], target: list[str]):
    '''Calculate the exact match ratio of the model output'''
    assert len(pred) == len(target)
    count = 0
    for i, _ in enumerate(pred):
        gt_str = target[i]
        pred_str = pred[i]
        if gt_str == pred_str:
            count += 1
    ratio = count/len(target)
    return {'exact_match': ratio}


def calculate_metrics(pred: list[str], target: list[str]):
    '''Calculate the rouge, bleu and exact match ratio of the model output'''
    remove_eos(pred)
    result = {}
    result.update(get_rouge(pred, target))
    result.update(get_bleu(pred, target))
    result.update(exact_match(pred, target))
    return result


if __name__ == '__main__':
    seten_1 = ["Hello world"]
    senten_2 = ["Hello world"]
    print(calculate_metrics(seten_1, senten_2))
