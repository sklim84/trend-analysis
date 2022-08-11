import pandas as pd
import tomotopy as tp
from treform.topic_model.pyTextMinerTopicModel import pyTextMinerTopicModel

from _datasets import jpss_data
from topic_modeling.lda.commons import lda_model, get_topic_labeler

# 생성 토픽 수
topic_number = 10
# 기존 생성한 모델 재사용여부
reuse_trained_model = False

model = None
if reuse_trained_model:
    # 모델 로드
    model = tp.DMRModel.load('./models/jpss.model')
else:
    # 데이터 로드 및 전처리
    timestamps, dataset = jpss_data.load_for_topic(timestamp_index=1, target_index=3)

    # DMR 모델 학습 및 저장
    model = lda_model(dataset, topic_number)
    model.save('./models/jpss.model', True)

# document별 dominant topic 정보 저장
topic_model = pyTextMinerTopicModel()
df_topic_sents_keywords, matrix = topic_model.format_topics_sentences(topic_number=10, mdl=model)
# formatting
df_dominant_topic_for_each_doc = df_topic_sents_keywords.reset_index()
df_dominant_topic_for_each_doc.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic_for_each_doc.to_csv('./results/jpss_dominent_topic_for_each_doc.csv', index=False,
                                      encoding='utf-8-sig')

# topic label, keyword 저장
labeler = get_topic_labeler(model)
df_topic_label_keyword = pd.DataFrame(columns=['topic number', 'label', 'keywords'])
for index, topic_number in enumerate(range(model.k)):
    label = ' '.join(label for label, score in labeler.get_topic_labels(topic_number, top_n=5))
    keywords = ' '.join(keyword for keyword, prob in model.get_topic_words(topic_number))
    df_topic_label_keyword.loc[index] = [topic_number, label, keywords]
df_topic_label_keyword.to_csv('./results/jpss_topic_label_keyword.csv', index=False, encoding='utf-8-sig')
