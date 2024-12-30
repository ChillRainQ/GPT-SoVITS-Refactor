import os.path

from tools import file_util

CURRENT_DIR = os.path.dirname(__file__)
GPT_DIR = os.path.join(CURRENT_DIR, "gpt")
SOVITS_DIR = os.path.join(CURRENT_DIR, "sovits")
CNHUBERT_PATH = os.path.join(CURRENT_DIR, "chinese-hubert-base")
BERT_PATH = os.path.join(CURRENT_DIR, "chinese-roberta-wwm-ext-large")

GPT_LIST = file_util.get_files_in_dir(GPT_DIR)
SOVITS_LIST = file_util.get_files_in_dir(SOVITS_DIR)

if __name__ == "__main__":
    print(GPT_LIST)
    print(CNHUBERT_PATH)
    print(BERT_PATH)
    print(SOVITS_LIST)