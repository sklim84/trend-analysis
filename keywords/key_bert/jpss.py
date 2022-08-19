from keybert import KeyBERT

from _datasets import jpss_data

####################
# Journal of Payments Strategy & Systems 논문
# - Format : id(eid) | year | title | abstract | keywords
# - 기간 : 2017-2021
# - 분석대상 : abstract
####################

# 데이터 로드 및 전처리
dataset = jpss_data.load_for_keyword(target_index=3, reuse_preproc=False)

# KeyBERT 기반 Keyword 추출
model = KeyBERT()
documents = ' '.join(dataset)
keywords = model.extract_keywords(documents, top_n=100, keyphrase_ngram_range=(1, 1))

with open('./results/jpss_keywords.txt', 'w', encoding='utf-8') as fout:
    for word, r in keywords:
        print('{}\t{}\n'.format(word, r))
        fout.write('{}\t{}\n'.format(word, r))
fout.close()
