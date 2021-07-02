import re
import os
import pandas as pd
from tqdm import tqdm


def preprocessing(sentence: str, lang="gy"):
    if lang in ["gang", "jeon"]:
        try:
            sentence = sentence.split("\t")[1].strip()
        except IndexError:
            sentence = sentence
    elif lang == "jeju":
        sentence = re.sub('^JJ\w+', "", sentence).strip()

    sentence = sentence.replace("(())", "")
    sentence = re.sub('^\d:', "", sentence)
    sentence = re.sub('(@\w+)', "", sentence)
    sentence = sentence.replace("  ", "")
    return sentence.strip()


def split_dialect(sentence: str):
    dialect = re.sub(r"/\([^)]*\)", "", sentence)
    korean = re.sub(r"\([^)]*\)/", "", sentence)

    dialect = re.sub("[()]", "", dialect)
    korean = re.sub("[()]", "", korean)
    return korean, dialect


if __name__ == "__main__":
    dialects = {"전라도": "jeon", "제주도": "jeju", "강원도": "gang", "경상도": "gy", "충청도": "chung"}
    for dialect in dialects.keys():
        path = "D:/Dataset/한국어 방언 발화 데이터({})/Training/[라벨]{}_학습데이터_1/".format(dialect, dialect)
        lst = [i for i in os.listdir(path) if i.endswith(".txt")]
        result = []
        for i in tqdm(lst):
            with open(path + i, "r", encoding="utf-8-sig") as f:
                for sen in f.readlines():
                    sen = preprocessing(sen, dialects[dialect])
                    if sen == "":
                        continue
                    korean_sen, dialect_sen = split_dialect(sen.strip())
                    result.append([korean_sen, dialect_sen])
        df = pd.DataFrame(result, columns=["표준어", dialect])
        df.to_csv("./{}_data.csv".format(dialects[dialect]), index=False, encoding="utf-8-sig")
