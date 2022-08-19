import treform.keyword.textrank as tr

from _datasets import news_data

####################
# 네이버 뉴스
# - Format : date | press | title | link | content
# - 기간 : 2017-2021
# - 총 8119건
# - 분석대상 : content
####################

# 데이터 로드 및 전처리
dataset = news_data.load_for_keyword(target_index=4, reuse_preproc=True)

# TextRank 기반 Keyword 추출
keyword_extractor = tr.TextRank(pos_tagger_name='mecab', mecab_path='C:\\mecab\\mecab-ko-dic', lang='ko')
keyword_extractor.build_keywords(' '.join(dataset))
keywords = keyword_extractor.get_keywords(limit=100)

with open('./results/news_keywords.txt', 'w', encoding='utf-8') as fout:
    for word, r in keywords:
        print('{}\t{}\n'.format(word, r))
        fout.write('{}\t{}\n'.format(word, r))
fout.close()
