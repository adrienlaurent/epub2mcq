# Copyright Adrien Laurent 2022
# All rights reserved.

import transformers
from sense2vec import Sense2Vec
from better_profanity import profanity

import config


def load_distractor_models():
    s2v = Sense2Vec().from_disk("s2v_old")
    device_id = 0 if config.USE_GPU else -1
    fm_pipeline = transformers.pipeline(
        "fill-mask", model="distilbert-base-uncased", device=device_id
    )
    return s2v, fm_pipeline


def get_pos_tag(nlp, text, word):
    doc = nlp(text)
    for token in doc:
        if token.text.lower() == word.strip().lower():
            return token.pos_
    return None


def match_word_shape(word, shape):
    """
    Given a word and a spacy shape, return the word capitalized as shape
    (title, lower, or upper)
    Example:
        >>> match_word_shape("hello", "Xxxxx")
        "Hello"
        >>> match_word_shape("Apple", "xxxxx")
        "apple"
    """

    if shape.istitle():
        return word.capitalize()
    elif shape.islower():
        return word.lower()
    elif shape.isupper():
        return word.upper()
    else:
        return word


def _join_text_context(text, context_before, context_after):
    return " ".join(map(str.strip, [context_before, text, context_after])).strip()


def bert_distractors(pipeline, nlp, word, text, context_before, context_after):
    """BERT"""
    orig_text = text
    orig_word = word
    text = text.lower()
    word = word.lower().strip()

    # multiple words are not supported
    if text.count(word) > 1:
        return []

    orig_pos_tag = get_pos_tag(nlp, orig_text, orig_word)
    orig_lemma = nlp(word)[0].lemma_

    text = text.replace(word, "[MASK]")

    text_with_context = _join_text_context(text, context_before, context_after)
    preds = pipeline(text_with_context, top_k=50)

    distractors = []

    for p in preds:
        sent = p["sequence"]
        pred_word = p["token_str"].strip()
        score = p["score"]

        # ignore if the score is too low or too high
        if not 0.001 < score < 0.1:
            continue

        # make sure the predicted word has the same POS tag as the original
        pred_pos_tag = get_pos_tag(nlp, sent, pred_word)
        if pred_pos_tag != orig_pos_tag:
            continue

        # make sure the predicted word has different lemma than the original
        pred_lemma = nlp(pred_word)[0].lemma_
        if pred_lemma == orig_lemma:
            continue

        # make sure the predicted word is not the original word
        if pred_word.lower() == word.strip():
            continue

        # make sure it's not profanity
        if profanity.contains_profanity(pred_word):
            continue

        # capitalize if it's a proper noun
        if orig_pos_tag == "PROPN":
            pred_word = pred_word.capitalize()

        distractors.append(pred_word)

    return distractors


def s2v_distractors(s2v, word):
    """Sense2Vec"""
    res = []
    word = word.lower()
    word = word.replace(" ", "_")

    try:
        sense = s2v.get_best_sense(word)
        most_similar = s2v.most_similar(sense, n=20)
    except TypeError:
        return res

    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ").lower()

        # make sure it's not profanity
        if profanity.contains_profanity(append_word):
            continue

        if append_word.lower() != word:
            res.append(append_word.strip())

    return res


def get_distractors(
    keyword, text, context_before, context_after, nlp, fm_pipeline, s2v, top_n
):
    """
    Args:
        keyword (str): the keyword
        text (str): the sentence(s)
        context_before (str): the context before the text
        context_after (str): the context after the text
        nlp: spacy model
        fm_pipeline: the Fill-Mask model
        s2v: the Sense2Vec model
        n_distractors (int): the number of distractors to return
    """

    text_with_context = _join_text_context(text, context_before, context_after)
    distractors = set()

    # get same POS distractors from text & context
    kw_pos_tag = get_pos_tag(nlp, text_with_context, keyword)
    for t in nlp(text_with_context):
        if t.text.lower().strip() == keyword.lower().strip():
            continue
        if t.pos_ == kw_pos_tag:
            distractors.add(t.text.strip())

    distractors = list(distractors)[:2]

    if len(distractors) < top_n:
        distractors.extend(
            bert_distractors(
                fm_pipeline, nlp, keyword, text, context_before, context_after
            )
        )
    if len(distractors) < top_n:
        distractors.extend(s2v_distractors(s2v, keyword))

    return list(dict.fromkeys(distractors))[:top_n]
