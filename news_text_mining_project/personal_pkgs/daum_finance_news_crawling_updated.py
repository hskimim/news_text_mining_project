
# coding: utf-8

# In[1]:


import requests
from scrapy.http import TextResponse
import pandas as pd
import datetime


# In[2]:


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


# In[3]:


len(title_ls) == len(link_ls)


# In[4]:


df = pd.DataFrame()
df['title'] = title_ls
df['link'] = link_ls
df['link'] = ['http://m.finance.daum.net' + i for i in df['link']]
df.tail()


# In[5]:


content_ls = []

for i in df['link'].values:
    req = requests.get(i)
    response = TextResponse(req.url , body=req.text , encoding='utf-8')
    content_ls.append(','.join(response.xpath('//*[@id="dmcfContents"]/section/p/text()').extract()).replace(',',''))


# In[6]:


len(title_ls) == len(link_ls) == len(content_ls)


# In[7]:


df['content'] = content_ls
df['content'] = [i.replace("\t",'').replace('\n','') for i in df['content'].values]
df.tail()


# In[8]:


df.to_csv('../folder/'+'{}_with_link_daum.csv'.format(str(datetime.datetime.now().month) + str(datetime.datetime.now().day),index=False),index=False)

