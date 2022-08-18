import treform.keyword.textrank as tr

from _datasets import news_data
from keybert import KeyBERT

####################
# 네이버 뉴스
# - Format : date | press | title | link | content
# - 기간 : 2017-2021
# - 총 8119건
# - 분석대상 : content
####################

# 데이터 로드 및 전처리
dataset = news_data.load_for_keyword(target_index=4, reuse_preproc=True)

# KeyBERT 기반 Keyword 추출
model = KeyBERT('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
keywords = model.extract_keywords(' '.join(dataset), top_n=100, keyphrase_ngram_range=(1, 1))

with open('./results/kpsa_keywords.txt', 'w', encoding='utf-8') as fout:
    for word, r in keywords:
        print('{}\t{}\n'.format(word, r))
        fout.write('{}\t{}\n'.format(word, r))
fout.close()
