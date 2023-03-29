from summarize import load_summarization_model_tokenizer, summarizer
from questions import load_model,get_question
from keywords import *
from distractors import *
import fileinput
import config
from mcq import mcq_generator
from epub import get_chapters_from_epub

#loading models
summarize_model, summarize_tokenizer = load_summarization_model_tokenizer()
#questions_model, questions_tokenizer = load_model()
#s2v, sentencemodel, normalized_levenshtein = load_distractor_model()

chapters = get_chapters_from_epub('book.epub') 
chunk = chapters[3]["chunks"][6]
chunk = " ".join(filter(None, chunk.split()))

print(chunk)

summary = summarizer(chunk, summarize_model, summarize_tokenizer)

print("================")

print(summary)

print("================")


from transformers import pipeline
summarizer = pipeline("summarization")
print(summarizer(chunk, min_length=50, max_length=100))