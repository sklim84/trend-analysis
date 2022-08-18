from _datasets import kpsa_data
from keywords.commons import execute_KRWordRank
import treform as ptm
import treform.keyword.textrank as tr
from keybert import KeyBERT

####################
# 지급결제학회 논문
# - Format : id(doi) | year | title | abstract | keywords
# - 기간 : 2017-2021
# - 분석대상 : abstract
####################

# 데이터 로드 및 전처리
dataset = kpsa_data.load_for_keyword(target_index=3, reuse_preproc=True)

####################
# KRWordRank 기반 Keyword 추출
# - 파라미터 : min_count=5, max_length=10, beta=0.85, max_iter=10
#  - GitHub에 공유된 KRWordRank 테스트 코드의 설정값
#  - https://github.com/lovit/KR-WordRank/blob/master/tests/test_krwordrank.py
####################

# KRWordRank
# keywords = execute_KRWordRank(dataset,
#                               min_count=5,  # 단어의 최소 출현 빈도수 (그래프 생성 시)
#                               max_length=10,  # 단어의 최대 길이
#                               beta=0.85,  # PageRank의 decaying factor beta
#                               max_iter=10,
#                               num_words=100)
# print(keywords)

# with open('./results/kpsa_keywords.txt', 'w', encoding='utf-8') as fout:
#     for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
#         print('{}\t{}\n'.format(word, r))
#         fout.write('{}\t{}\n'.format(word, r))
# fout.close()

# TextRank
# keyword_extractor = tr.TextRank(pos_tagger_name='mecab', mecab_path='C:\\mecab\\mecab-ko-dic', lang='ko')
# keyword_extractor.build_keywords(' '.join(dataset))
# keywords = keyword_extractor.get_keywords(limit=100)
#
# with open('./results/kpsa_keywords.txt', 'w', encoding='utf-8') as fout:
#     for word, r in keywords:
#         print('{}\t{}\n'.format(word, r))
#         fout.write('{}\t{}\n'.format(word, r))
# fout.close()

# KeyBert TODO save model
model = KeyBERT('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
keywords = model.extract_keywords(' '.join(dataset), top_n=100, keyphrase_ngram_range=(1, 1))
print(keywords)

with open('./results/kpsa_keywords.txt', 'w', encoding='utf-8') as fout:
    for word, r in keywords:
        print('{}\t{}\n'.format(word, r))
        fout.write('{}\t{}\n'.format(word, r))
fout.close()