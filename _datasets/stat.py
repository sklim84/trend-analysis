import pandas as pd
import plotly.express as px

pd.options.plotting.backend = "plotly"

df_jpss = pd.read_csv('jpss.csv')
df_kpsa = pd.read_csv('kpsa.csv')
df_news = pd.read_csv('news.csv')
df_news['date'] = pd.to_datetime(df_news['date'])

# 연도별 문서 수
df_jpss_cnt = df_jpss.groupby(['year'])['year'].count()
df_kpsa_cnt = df_kpsa.groupby(['year'])['year'].count()
df_news_cnt = df_news.groupby(df_news.date.dt.year)['date'].count()

df_cnt = pd.concat([df_jpss_cnt, df_kpsa_cnt, df_news_cnt], axis=1)
df_cnt.columns = ['jpss', 'kpsa', 'news']
df_cnt.index = df_cnt.index.astype(int)
fig = df_cnt.plot(kind='bar')
fig.write_html('./stat_count_by_year.html')

print(df_cnt)

