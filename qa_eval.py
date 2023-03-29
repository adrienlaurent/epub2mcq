import torch
import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


device = config.DEVICE


def load_qa_eval_model():
    device_id = 0 if config.USE_GPU else -1
    qa_pipe = pipeline("question-answering", device=device_id)
    model_name = "iarfmoose/bert-base-cased-qa-evaluator"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.eval()
    if config.USE_FP16:
        model = model.half()
    model = model.to(device)
    return model, tokenizer, qa_pipe


def score_qa_pair(text, question, answer, model, tokenizer, qa_pipe, nlp):
    """
    Args:
        text: str, the text to be scored
        question: str, the question to be scored
        answer: str, the answer to the question
        model: the model
        tokenizer: the tokenizer
        qa_pipe: the pipeline
        nlp: spacy model

    Returns:
        score: float, the score of the text
        better_answer: str, a better answer suggested by the QA model, or None
    """

    encoding = tokenizer(
        text=question,
        text_pair=answer,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model(
            input_ids=encoding.input_ids.to(device),
            token_type_ids=encoding.token_type_ids.to(device),
            attention_mask=encoding.attention_mask.to(device),
        )
        qa_eval_score = torch.sigmoid(output[0][0][1]).cpu().detach().item()

    # answer question using question answering model & compare to provided answer
    qa_res = qa_pipe(question=question, context=text)
    extracted_answer = qa_res["answer"].lower().strip()
    similarity = nlp(answer).similarity(nlp(extracted_answer))
    if similarity < 0:
        similarity = 0
    qa_match_score = similarity * qa_res["score"]

    score = qa_match_score * qa_eval_score

    better_answer = None

    # if the QA model is very confident, then propose its answer
    if (
        extracted_answer != answer.lower().strip()
        and len(extracted_answer.split()) <= 3
        and qa_res["score"] > 0.9
    ):
        better_answer = extracted_answer

    return score, better_answer