import gc
import torch
import config
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk import sent_tokenize


device = config.DEVICE


def load_model():
    question_models = {}
    question_tokenizers = {}

    if "valhalla" in config.QG_MODELS:
        model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl").eval()
        if config.USE_FP16:
            model = model.half()
        model = model.to(device)
        question_models["valhalla"] = model
        question_tokenizers["valhalla"] = T5Tokenizer.from_pretrained(
            "valhalla/t5-base-qg-hl"
        )

    if "ramsrigouthamg" in config.QG_MODELS:
        model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_squad_v1").eval()
        if config.USE_FP16:
            model = model.half()
        model = model.to(device)
        question_models["ramsrigouthamg"] = model
        question_tokenizers["ramsrigouthamg"] = T5Tokenizer.from_pretrained(
            "ramsrigouthamg/t5_squad_v1"
        )

    return question_models, question_tokenizers


def get_questions_valhalla(chunk, answer, model, tokenizer):
    sentences = sent_tokenize(chunk)
    highlighted_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()

        # ignore very short sentences
        if len(sentence) < 10:
            continue

        answer_idx = sentence.lower().find(answer.lower())

        if answer_idx != -1:
            highlighted_chunk = (
                "generate question: "
                + sentence[:answer_idx]
                + " <hl> "
                + answer
                + " <hl>"
                + sentence[answer_idx + len(answer) :]
            )
            break

    # not found or sentence too short: do not generate questions
    if len(highlighted_chunk) < 10:
        return []

    encoding = tokenizer(highlighted_chunk, truncation=True, return_tensors="pt")
    input_ids, attention_mask = (
        encoding["input_ids"].to(device),
        encoding["attention_mask"].to(device),
    )

    with torch.no_grad():
        outs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **config.QUESTION_GENERATE_PARAMS_valhalla
        )
    
    questions = [tokenizer.decode(out.cpu(), skip_special_tokens=True) for out in outs]
    return questions


def get_questions_ramsrigouthamg(chunk, answer, model, tokenizer):
    sentences = sent_tokenize(chunk)
    text = ""

    for sentence in sentences:
        sentence = sentence.strip()

        answer_idx = sentence.lower().find(answer.lower())

        if answer_idx != -1:
            text = "context: {} answer: {}".format(sentence, answer)
            break

    if len(text) < 10:
        return []

    encoding = tokenizer(
        text,
        max_length=config.QUESTION_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    with torch.no_grad():
        outs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **config.QUESTION_GENERATE_PARAMS_ramsrigouthamg
        )
    
    questions = [
        tokenizer.decode(out.cpu(), skip_special_tokens=True)
        .replace("question:", "")
        .strip()
        for out in outs
    ]
    return questions


def get_questions(chunk, answer, models, tokenizers):
    question_dispatcher = {
        "valhalla": get_questions_valhalla,
        "ramsrigouthamg": get_questions_ramsrigouthamg,
    }
    all_questions = set()
    for key in config.QG_MODELS:
        questions = question_dispatcher[key](
            chunk, answer, models[key], tokenizers[key]
        )
        for q in questions:
            all_questions.add(q.strip())

    return list(all_questions)
