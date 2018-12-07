# coding: utf-8
import sys
sys.path.append('../../personal_pkgs')

from korean_crawling_bundles import *

from konlpy.tag import *
okt = Okt()
import datetime
from IPython.display import display

import re
import numpy as np
import pandas as pd
from pprint import pprint
from konlpy.tag import Hannanum

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
from pykospacing import spacing

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import telegram
import pickle
### packages for crawling
from scrapy.http import TextResponse
import requests
import datetime

### naver crwaling is started!

print('naver crawling is started!')
title_ls,link_ls = [],[]
content_str = 0
for page in range(1,30+1):
    url = 'https://m.news.naver.com/newsflash.nhn?mode=LS2D&sid1=101&sid2=259&page={}'.format(page)
    req = requests.get(url)
    response = TextResponse(req.url , body=req.text , encoding='utf-8')
    for content in range(1,20+1):
        try:
            content_str += 1
            title_ls.append(response.xpath('//*[@id="newsflash{}"]/a/div/span[1]/strong/text()'.format(content_str)).extract()[0])
            link_ls.append(response.xpath('//*[@id="newsflash{}"]/a/@href'.format(content_str)).extract()[0])
        except : pass

df = pd.DataFrame()
df['title'] = title_ls
df['link'] = link_ls
df['link'] = ['https://m.news.naver.com' + i for i in df['link']]

content_ls = []

for i in df['link'].values:
    req = requests.get(i)
    response = TextResponse(req.url , body=req.text , encoding='utf-8')
    content_ls.append(','.join(response.xpath('//*[@id="dic_area"]/text()').extract()).replace(',',''))

df['content'] = content_ls
df['content'] = [i.replace("\t",'').replace('\n','') for i in df['content'].values]

df.to_csv('../folder/'+'{}_with_link_naver.csv'.format(str(datetime.datetime.now().month) + str(datetime.datetime.now().day),index=False),index=False)
### finish naver news crawling

print('daum crawling is started!')
title_ls,link_ls = [],[]
for page in range(1,21+1):
    url = 'http://m.finance.daum.net/m/news/news_list.daum?type=main&page={}'.format(page)
    req = requests.get(url)
    response = TextResponse(req.url , body=req.text , encoding='utf-8')
    for content in range(1,10+1):
        try :
            title_ls.append(response.xpath('//*[@id="mArticle"]/div[2]/ul/li[{}]/a/strong/text()'.format(content)).extract()[0])
            link_ls.append(response.xpath('//*[@id="mArticle"]/div[2]/ul/li[{}]/a/@href'.format(content)).extract()[0])
        except : pass

df = pd.DataFrame()
df['title'] = title_ls
df['link'] = link_ls
df['link'] = ['http://m.finance.daum.net' + i for i in df['link']]

content_ls = []

for i in df['link'].values:
    req = requests.get(i)
    response = TextResponse(req.url , body=req.text , encoding='utf-8')
    content_ls.append(','.join(response.xpath('//*[@id="dmcfContents"]/section/p/text()').extract()).replace(',',''))

df['content'] = content_ls
df['content'] = [i.replace("\t",'').replace('\n','') for i in df['content'].values]

df.to_csv('../folder/'+'{}_with_link_daum.csv'.format(str(datetime.datetime.now().month) + str(datetime.datetime.now().day),index=False),index=False)

### finish daum crawling
print('## LDA process is started from now!')
### finish crawling process and start LDA process

# stopwords는 불필요한 단어, 즉 조사나 관사들을 없애는 툴이다.

stop_words = list(pd.read_csv('kor_stop_words.csv').T.iloc[:1,:].values[0])
stop_words[:10]


print('## overall dataframe is merged now')
naver_df = pd.read_csv('../folder/'+'{}_with_link_naver.csv'.format(str(datetime.datetime.now().month) + str(datetime.datetime.now().day),index=False))
daum_df = pd.read_csv('../folder/'+'{}_with_link_daum.csv'.format(str(datetime.datetime.now().month) + str(datetime.datetime.now().day),index=False))
daum_df = daum_df.loc[:,['title','link','content']]

df = pd.concat([naver_df,daum_df],axis=0)
df.reset_index(drop=True,inplace=True)

# # 병합한 데이터 프레임에서 중복되는 기사 내용 없애주기
append_ls,append_idx = [],[]
for idx,i in enumerate(df['title']) :
    if i not in append_ls:
        append_ls.append(i)
        append_idx.append(idx)

df = df.loc[append_idx]

df.fillna('0',inplace=True)
#결측치를 0으로 바꿔준다.

df['content'] = [i.replace('  ','') for i in df['content'].values]

# # Tuning_process
# Convert to list
data = df.content.values.tolist()

print("tuning process is operating")

# Remove Emails
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

# 뒤에 부분의 저자에 관한 내용을 가능한 만큼 잘라준다.
data = [val[:re.search('Copyrights|ⓒ|▶',val).start()] if ('Copyrights' in val) or ('ⓒ' in val) or ("▶" in val) else val for val in data]

# 대괄호 안에 저자의 이름이 나와있는 경우를 없애준다.
data = [re.sub('\[.+\]','',val) for val in data]

warnings.filterwarnings("error")

spacing_ls = []

print('### tuning about spacing is operating becuz of spacy and okt tuning process, it\'ll take over than 5min. ')
for idx,val in enumerate(data) :
    try :
         spacing_ls.append(','.join([spacing(val) for val in val.split(".")]).replace(",",''))
    except : spacing_ls.append(val)

data = spacing_ls.copy()


# # tokenization process

# - 정규식 표현을 통해서 문장 내에 이메일과 기타 특수 문자들을 없애주었지만, 여전히 난잡해보인다.
# - LDA 알고리즘을 사용하기 위해서는, 문장들을 단어들의 묶음으로 변환시켜주는 과정이 필요하다.
# - 이러한 과정을 Tokenization 이라고 한다.

data_words = [okt.nouns(val) for val in data]

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        # deacc=True removes punctuations
        # 구두점(말끝에 찍는 쉼표나 점들을 의미) 을 없애주는 것이다.

gensim_data_words = list(sent_to_words(data))

warnings.filterwarnings("ignore")

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

data_lemmatized = data_words_bigrams.copy()

print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

print("# Build LDA model now")

mallet_path = '/home/hskimim/Documents/mallet-2.0.8/bin/mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=5, id2word=id2word)

def format_topics_sentences(ldamodel, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamallet , corpus=corpus, texts=data)
# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

recommendation_df = df_dominant_topic.sort_values(by='Topic_Perc_Contrib',ascending=False)
recommendation_df.reset_index(drop=True,inplace=True)
#recommendation_df 에서 중복되는 기사를 없애주자.
duplicate_ls = []
duplicate_idx = []
for idx,val in enumerate(recommendation_df['Dominant_Topic'].values):
    if val not in duplicate_ls:
        duplicate_ls.append(val)
        duplicate_idx.append(idx)
        if len(duplicate_ls) == 5 : break


print('recommendation_df is gonna be made and overall process is done')

recommendation_df = df.loc[recommendation_df.loc[duplicate_idx]['Document_No'].values].iloc[:,:2]

# my_token = pickle.load( open('pw.pickle', 'rb'))
# my_token = my_token
#
# bot = telegram.Bot(token = my_token) #봇을 생성합니다.
#
# chat_id = bot.getUpdates()[-1].message.chat.id
#
# bot.sendMessage(chat_id=chat_id, text='안녕하세요. 좋은 아침입니다. {}토픽 별 기사 5개 추천해드리겠습니다.'.format(\
# str(datetime.datetime.now().month)+' 월 ' + str(datetime.datetime.now().day)+' 일 '))
#
# for news in recommendation_df['link'].tolist() :
#     bot.sendMessage(chat_id = chat_id, text=news)
#
my_token = '717292107:AAGuwGQIlMnr-LlhtH5xqR4Zud8GyyKXbQY' # 토큰을 변수에 저장합니다.

bot = telegram.Bot(token = my_token) # bot을 선언합니다.

chat_id = bot.getUpdates()[-1].message.chat.id

bot.sendMessage(chat_id=768469872, text='안녕하세요. 좋은 아침입니다. {}토픽 별 기사 5개 추천해드리겠습니다.'.format(\
str(datetime.datetime.now().month)+' 월 ' + str(datetime.datetime.now().day)+' 일 '))

for news in recommendation_df['link'].tolist() :
    bot.sendMessage(chat_id = 768469872, text=news)
