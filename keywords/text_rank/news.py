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
doc_group_by_time = news_data.load_for_keyword(timestamp_name='date', target_name='content',
                                               timestamp_format='%Y-%m-%d',
                                               timestamp_group_format='Y', reuse_preproc=True)

# TextRank 기반 Keyword 추출
for timestamp in doc_group_by_time.keys():
    print(timestamp)
    documents = doc_group_by_time[timestamp]
    keyword_extractor = tr.TextRank(pos_tagger_name='mecab', mecab_path='C:\\mecab\\mecab-ko-dic', lang='ko')
    keyword_extractor.build_keywords(' '.join(documents))
    keywords = keyword_extractor.get_keywords(limit=10)

    with open('./results/{}_news_keywords.txt'.format(timestamp.strftime('%Y')), 'w', encoding='utf-8') as fout:
        for word, r in keywords:
            print('{}\t{}\n'.format(word, r))
            fout.write('{}\t{}\n'.format(word, r))
    fout.close()
