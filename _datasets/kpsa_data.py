import pathlib
import pickle
import re

import pandas as pd
import treform as ptm


####################
# 지급결제학회 논문(2017~2021) abstract 분석
# - Format : id(doi) | year | title | abstract | keywords
# - 기간 : 2017-2021
# - 분석대상 : abstract
####################

####################
# 데이터 로드 및 전처리
# - 전처리 : Tokenizing, POS tagging & filtering, n-gram, Stopword Filtering
####################


def load_for_keyword(timestamp_name, target_name, timestamp_format='%Y', timestamp_group_format='Y',
                     reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'kpsa.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'kpsa_pp_for_keyword.pkl', 'rb') as fin:
            doc_group_by_time = pickle.load(fin)
        fin.close()
        return doc_group_by_time

    # 데이터 로드
    df_jpss = pd.read_csv(loc_data)
    df_jpss[timestamp_name] = pd.to_datetime(df_jpss[timestamp_name], format=timestamp_format)
    df_jpss.set_index(timestamp_name, inplace=True)

    # timestamp별 그룹화
    doc_group_by_time = {time: group for time, group in df_jpss.groupby(pd.Grouper(freq=timestamp_group_format))}

    for time in doc_group_by_time.keys():
        target = doc_group_by_time[time][target_name].astype(str).values.tolist()

        # 전처리
        pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                                ptm.tokenizer.MeCab('/home/sklim/mecab-ko-dic-2.1.1-20180720/'),
                                ptm.helper.POSFilter('NN*'),
                                ptm.helper.SelectWordOnly(),
                                ptm.ngram.NGramTokenizer(1, 1),
                                ptm.helper.StopwordFilter(file=loc_stopwords))
        result = pipeline.processCorpus(target)

        documents = []
        for doc in result:
            document = []
            for sent in doc:
                new_sent = ' '.join(sent)
                new_sent = re.sub('[^A-Za-z0-9가-힣_ ]+', '', new_sent)
                new_sent = new_sent.strip()
                document.append(new_sent)
            documents.append(' '.join(document))

        doc_group_by_time[time] = documents

    # 전처리된 결과 저장
    with open(here / 'kpsa_pp_for_keyword.pkl', 'wb') as fout:
        pickle.dump(doc_group_by_time, fout)
    fout.close()

    return doc_group_by_time


def load_for_coword(target_index, reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'kpsa.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'kpsa_pp_for_coword.pkl', 'rb') as fin:
            documents = pickle.load(fin)
        fin.close()
        return documents

    # 데이터 로드
    df_kpsa = pd.read_csv(loc_data)
    target = df_kpsa.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('/home/sklim/mecab-ko-dic-2.1.1-20180720/'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),
                            ptm.ngram.NGramTokenizer(1, 1),
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    result = pipeline.processCorpus(target)

    # 구조 변경 : Sentence co-occurrence word를 찾기 위해 하나의 setence를 하나의 document로 변경
    documents = []
    for doc in result:
        for sent in doc:
            new_sent = ' '.join(sent)
            new_sent = re.sub('[^A-Za-z0-9가-힣_ ]+', '', new_sent)
            new_sent = new_sent.strip()
            if len(new_sent) > 0:
                documents.append(new_sent)

    # 전처리된 결과 저장
    with open(here / 'kpsa_pp_for_coword.pkl', 'wb') as fout:
        pickle.dump(documents, fout)
    fout.close()

    return documents


####################
# Term Weighting을 위한 데이터 로드 및 처리
# - 전처리 : Tokenizing, POS tagging & filtering, Stopword Filtering
####################
def load_for_term_weighting(label_index, target_index):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'kpsa.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 데이터 로드
    label = ptm.CorpusFromCSVFile(loc_data, label_index)
    content = ptm.CorpusFromCSVFile(loc_data, target_index)
    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('/home/sklim/mecab-ko-dic-2.1.1-20180720/'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    return label.docs[1:], pipeline.processCorpus(content.docs[1:])


def load_for_topic(timestamp_index, target_index, reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'kpsa.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'kpsa_pp_for_topic.pkl', 'rb') as fin:
            timestamps = pickle.load(fin)
            documents = pickle.load(fin)
        fin.close()
        return timestamps, documents

    # 데이터 로드
    df_kpsa = pd.read_csv(loc_data)
    timestamps = df_kpsa.iloc[:, timestamp_index].tolist()
    target = df_kpsa.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('/home/sklim/mecab-ko-dic-2.1.1-20180720/'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    result = pipeline.processCorpus(target)

    documents = []
    for doc in result:
        document = []
        for sent in doc:
            for word in sent:
                new_word = re.sub('[^A-Za-z0-9가-힣_ ]+', '', word)
                new_word = new_word.strip()
                if len(new_word) > 0:
                    document.append(new_word)
        documents.append(document)

    # 전처리된 결과 저장
    with open(here / 'kpsa_pp_for_topic.pkl', 'wb') as fout:
        pickle.dump(timestamps, fout)
        pickle.dump(documents, fout)
    fout.close()

    return timestamps, documents


def load_for_bertopic(timestamp_index, target_index, reuse_preproc=False):
    here = pathlib.Path(__file__).resolve().parent
    loc_data = here / 'kpsa.csv'
    loc_stopwords = here / 'stopwordsKor.txt'

    # 기전처리된 파일 사용 시
    if reuse_preproc:
        with open(here / 'kpsa_pp_for_bert.pkl', 'rb') as fin:
            timestamps = pickle.load(fin)
            documents = pickle.load(fin)
        fin.close()
        return timestamps, documents

    # 데이터 로드
    df_kpsa = pd.read_csv(loc_data)
    timestamps = df_kpsa.iloc[:, timestamp_index].tolist()
    target = df_kpsa.iloc[:, [target_index]].astype(str).values.tolist()
    # [[document1], ...] → [document1, ...]
    target = sum(target, [])

    # 전처리
    pipeline = ptm.Pipeline(ptm.splitter.NLTK(),
                            ptm.tokenizer.MeCab('/home/sklim/mecab-ko-dic-2.1.1-20180720/'),
                            ptm.helper.POSFilter('NN*'),
                            ptm.helper.SelectWordOnly(),  # 품사 태크 제거
                            ptm.helper.StopwordFilter(file=loc_stopwords))
    result = pipeline.processCorpus(target)

    documents = []
    for doc in result:
        document = ''
        for sent in doc:
            for word in sent:
                new_word = re.sub('[^A-Za-z0-9가-힣_ ]+', '', word)
                new_word = new_word.strip()
                if len(new_word) > 0:
                    document = ' '.join([document, new_word])
        documents.append(document)

    # 전처리된 결과 저장
    with open(here / 'kpsa_pp_for_bert.pkl', 'wb') as fout:
        pickle.dump(timestamps, fout)
        pickle.dump(documents, fout)
    fout.close()

    return timestamps, documents
