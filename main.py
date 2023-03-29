import spacy
from keybert import KeyBERT

from questions import load_model
from distractors import load_distractor_models
from mcq import mcq_generator
from qa_eval import load_qa_eval_model

import firebase

firebase.init_firebase()
ref = firebase.db.reference('books')

bookname = 'BOOK'

chapters = ref.child(bookname+'/sourceChapters').get()

# loading models
questions_model, questions_tokenizer = load_model()
s2v, fm_pipeline = load_distractor_models()
qa_eval_model, qa_eval_tokenizer, qa_pipe = load_qa_eval_model()
kw_model = KeyBERT(model="paraphrase-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_md")

#f = open("output.json", "a", encoding="utf-8")
#f.write("[\n")

chunk_and_mcqs = []
for num_chapter, one_chapter in enumerate(chapters):
    print ("\n\n\n ******************** CHAPTER "+str(num_chapter)+" *********************** \n")

    #skipping chapters this way to get the correct chapters ID#
    if num_chapter < 0 or num_chapter > 100 :
        continue

    for num_chunk, one_chunk in enumerate(one_chapter["chunks"]):
        print("\n ****** CONTEXT (CHUNK) "+str(num_chapter)+"/"+str(num_chunk)+" *********:\n " + str(one_chunk))

        # get the context
        context_before = one_chapter["chunks"][num_chunk-1] if num_chunk > 0 else ""
        context_after = one_chapter["chunks"][num_chunk+1] if num_chunk < len(one_chapter["chunks"])-1 else ""

        chunk_ref = ref.child(bookname+'/chapters').child(str(num_chapter)).child("chunks").child(str(num_chunk))
        #if chunk_ref.get() is not None:
        #    print("\n ****** MCQS ALREADY EXIST, SKIPPING CHUNK ******\n")
        #else:
        chunk_ref.set({
            "context": one_chunk,
            "mcqs": mcq_generator(
                one_chunk,
                context_before,
                context_after,
                questions_model,
                questions_tokenizer,
                fm_pipeline,
                s2v,
                kw_model,
                nlp,
                qa_eval_model,
                qa_eval_tokenizer,
                qa_pipe
            ),
        })
        

        print("\n")
    print("\n\n\n\n\n")
