class Config:
    NB_EPOCHS = 4
    LR = 1e-5
    MAX_LEN = 250
    N_SPLITS = 5
    TRAIN_BS = 32
    VALID_BS = 64
    MODEL_NAME = 'xlm-roberta-large'
    FILE_NAME = '../input/commonlitreadabilityprize/train.csv'
    TOKENIZER = transformers.XLMRobertaTokenizer.from_pretrained(MODEL_NAME)