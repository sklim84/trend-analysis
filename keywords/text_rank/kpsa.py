import treform.keyword.textrank as tr

from _datasets import kpsa_data

####################
# 지급결제학회 논문
# - Format : id(doi) | year | title | abstract | keywords
# - 기간 : 2017-2021
# - 분석대상 : abstract
####################

# 데이터 로드 및 전처리
dataset = kpsa_data.load_for_keyword(target_index=3, reuse_preproc=False)

# TextRank 기반 Keyword 추출
keyword_extractor = tr.TextRank(pos_tagger_name='mecab', mecab_path='C:\\mecab\\mecab-ko-dic', lang='ko')
keyword_extractor.build_keywords(' '.join(dataset))
keywords = keyword_extractor.get_keywords(limit=100)

with open('./results/kpsa_keywords.txt', 'w', encoding='utf-8') as fout:
    for word, r in keywords:
        print('{}\t{}\n'.format(word, r))
        fout.write('{}\t{}\n'.format(word, r))
fout.close()
