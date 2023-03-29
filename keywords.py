from flashtext import KeywordProcessor
import traceback
import pke
import string
from nltk.corpus import stopwords
import nltk
import argparse
from argparse import ArgumentParser
import config


def load_nltk():
    nltk.download("stopwords")


def get_nouns_multipartite(content):
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()

        extractor.load_document(input=content)
        #    not contain punctuation marks or stopwords as candidates.
        pos = {"PROPN", "NOUN"}
        # pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ["-lrb-", "-rrb-", "-lcb-", "-rcb-", "-lsb-", "-rsb-"]
        stoplist += stopwords.words("english")
        extractor.candidate_selection(pos=pos, stoplist=stoplist)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(**config.KEYWORD_GENERATE_PARAMS)
        keyphrases = extractor.get_n_best(n=config.KEYWORD_N_BEST)

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out


def get_keywords(originaltext, summarytext):
    keywords = get_nouns_multipartite(originaltext)
    #    print("keywords unsummarized: ", keywords)
    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword)

    keywords_found = keyword_processor.extract_keywords(summarytext)
    keywords_found = list(set(keywords_found))
    #    print("keywords_found in summarized: ", keywords_found)

    important_keywords = []
    for keyword in keywords:
        if keyword in keywords_found:
            important_keywords.append(keyword)

    return important_keywords[: config.KEYWORD_QT]


def get_keywords_v2(kw_model, nlp, text):
    res = set()
    doc = nlp(text)

    # extract KeyBERT keywords
    kb_res = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words="english",
        top_n=config.KEYBERT_TOP_N,
    )
    keybert_kws = [k[0] for k in kb_res]
    for kw in keybert_kws:
        res.add(kw.strip().lower())

    # extract named entities
    ner_kws = [ent.text for ent in doc.ents]
    for kw in ner_kws:
        res.add(kw.strip().lower())

    return list(res)
