import torch
import random
import numpy as np
import config
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer


device = config.DEVICE


def load_summarization_model_tokenizer():
    summary_model = T5ForConditionalGeneration.from_pretrained("t5-base").eval()
    summary_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    summary_model = summary_model.to(device)
    return summary_model, summary_tokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(config.SUMMARY_SEED)


def postprocesstext(content):
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final


def summarizer(text, model, tokenizer):
    text = text.strip().replace("\n", " ")
    text = text.strip().replace("  ", " ")

    text = "summarize: " + text
    # print (text)
    encoding = tokenizer.encode_plus(
        text,
        max_length=config.SUMMARY_MAX_LEN,
        pad_to_max_length=False,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = (
        encoding["input_ids"].to(device),
        encoding["attention_mask"].to(device),
    )

    with torch.no_grad():
      outs = model.generate(
          input_ids=input_ids,
          attention_mask=attention_mask,
          **config.SUMMARY_GENERATE_PARAMS
      )

    dec = [tokenizer.decode(ids.cpu().detach(), skip_special_tokens=True) for ids in outs]
    summary = dec[0]
    summary = postprocesstext(summary)
    summary = summary.strip()

    return summary


# summarized_text = summarizer(text,summary_model,summary_tokenizer)
# print (type(summarized_text))
