# Copyright Adrien Laurent 2022
# All rights reserved.

from questions import get_questions
from keywords import *
from distractors import *
from qa_eval import score_qa_pair
from utils import filter_text
import config


def mcq_generator(
    text,
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
    qa_pipe,
):
    filtered_text = filter_text(text)
    keywords_list = get_keywords_v2(kw_model, nlp, filtered_text)
    mcqs = []
    for keyword in keywords_list:

        print("\n\nKEYWORD: " + keyword)

        questions = get_questions(
            filtered_text, keyword, questions_model, questions_tokenizer
        )

        distractors = get_distractors(
            keyword,
            filtered_text,
            filter_text(context_before),
            filter_text(context_after),
            nlp,
            fm_pipeline,
            s2v,
            config.N_DISTRACTORS,
        )
        if not distractors:
            continue
        for question in questions:
            question = question.strip()

            print(" QUESTION: " + question)

            # ignore very short questions
            if len(question.strip()) < 10:
                continue

            # score the question-answer pair
            score, proposed_answer = score_qa_pair(
                filtered_text,
                question,
                keyword,
                qa_eval_model,
                qa_eval_tokenizer,
                qa_pipe,
                nlp,
            )

            # change answer if eval suggested better alternatives
            answer = proposed_answer if proposed_answer else keyword

            for oneDistractors in distractors:
                print("  DISTRACTORS: " + oneDistractors)

            # match capitalization between answer & distractors
            if all([d.islower() for d in distractors]):
                answer = answer.lower()
            if all([d.istitle() for d in distractors]):
                answer = answer.capitalize()

            # if eval suggested a better answer and it is a superset of the
            # original answer, add the missing parts to all distractors too
            if proposed_answer:
                if keyword.lower() in proposed_answer.lower():
                    distractors = [
                        proposed_answer.replace(keyword, d) for d in distractors
                    ]

            mcqs.append(
                {
                    "question": question,
                    "answer": answer,
                    "distractors": distractors,
                    "score": score,
                }
            )

    # sort mcqs based on score
    mcqs = sorted(mcqs, key=lambda x: x["score"], reverse=True)

    return mcqs
