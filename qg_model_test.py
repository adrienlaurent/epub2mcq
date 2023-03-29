# Copyright Adrien Laurent 2022
# All rights reserved.

import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl")


text = "generate question: lorem ipsum</s>"


inputs = tokenizer(text, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=4, max_length=32)

output = tokenizer.decode(output[0], skip_special_tokens=True)

print(output)