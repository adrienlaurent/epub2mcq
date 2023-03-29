# Copyright Adrien Laurent 2022
# All rights reserved.

import torch

# set this to either True or False
USE_GPU = True

# whether to use half-precision, set to False if models generate garbage
USE_FP16 = False

# checks for GPU and FP16 support
if USE_GPU and not torch.cuda.is_available():
    USE_GPU = False
    print(
        "Warning: You set the device to 'cuda' in config, but CUDA "
        "is not available. Using CPU instead"
    )
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
print("Using device:", DEVICE)

if USE_GPU is False and USE_FP16:
    USE_FP16 = False
    print(
        "Warning: Using FP16 on CPU is not supported, falling back to FP32 (full precision)"
    )
print("Using FP16:", USE_FP16)

N_DISTRACTORS = 4
LAMBDAVAL = 0.2
MIN_CHUNK_LEN = 30
MAX_CHUNK_LEN = 150

KEYBERT_TOP_N = 10

SUMMARY_MAX_LEN = 512
SUMMARY_SEED = 42
SUMMARY_GENERATE_PARAMS = {
    "early_stopping": True,
    "num_beams": 3,
    "num_return_sequences": 1,
    "no_repeat_ngram_size": 2,
    "min_length": 75,
    "max_length": 300
}

#QG_MODELS = ("valhalla", "ramsrigouthamg")
QG_MODELS = ("valhalla",)

QUESTION_MAX_LENGTH = 384
QUESTION_GENERATE_PARAMS_ramsrigouthamg = {
    "early_stopping": True,
    "max_length": 64,
    "num_beams": 8,
    "num_return_sequences": 1,
    "no_repeat_ngram_size": 2,
}
QUESTION_GENERATE_PARAMS_valhalla = {
    "early_stopping": True,
    "max_length": 64,
    "num_beams": 8,
    "num_return_sequences": 1,
    "no_repeat_ngram_size": 2,
}

KEYWORD_N_BEST = 15
KEYWORD_QT=10
KEYWORD_GENERATE_PARAMS = {
    "alpha": 1.1,
    "threshold": 0.75,
    "method": 'average'
}

DISTRACTORS_THRESHOLD = 0.6
DISTRACTORS_MAX_KEYWORD = 5