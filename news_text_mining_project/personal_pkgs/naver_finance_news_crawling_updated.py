
# coding: utf-8

# In[1]:


import pandas as pd
from scrapy.http import TextResponse
import requests
import datetime


# In[2]:


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
        except : print(page,content)


# In[3]:


len(title_ls) == len(link_ls)


# In[4]:


df = pd.DataFrame()
df['title'] = title_ls
df['link'] = link_ls
df['link'] = ['https://m.news.naver.com' + i for i in df['link']]
df.tail()


# In[5]:


content_ls = []

for i in df['link'].values:
    req = requests.get(i)
    response = TextResponse(req.url , body=req.text , encoding='utf-8')
    content_ls.append(','.join(response.xpath('//*[@id="dic_area"]/text()').extract()).replace(',',''))


# In[6]:


len(title_ls) == len(link_ls) == len(content_ls)


# In[7]:


df['content'] = content_ls
df['content'] = [i.replace("\t",'').replace('\n','') for i in df['content'].values]
df.tail()


# In[8]:


df.to_csv('../folder/'+'{}_with_link_naver.csv'.format(str(datetime.datetime.now().month) + str(datetime.datetime.now().day),index=False),index=False)

