import re
from nltk import sent_tokenize


def filter_text(text):
    sentences = list(map(str.strip, sent_tokenize(text.strip())))

    # remove page numbers
    sentences = list(map(lambda s: re.sub(r"\n[0-9]*\n", r" ", s), sentences))

    # remove "chapter N"
    chapter_re = re.compile(r"chapter [0-9]?")
    sentences = list(filter(lambda s: not chapter_re.match(s.lower()), sentences))

    # remove HTML tags
    sentences = list(map(lambda s: re.sub(r"<[^>]*>", r"", s), sentences))

    filtered_text = " ".join(sentences)
    return filtered_text