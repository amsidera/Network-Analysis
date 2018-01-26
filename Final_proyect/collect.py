# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 21:15:09 2017

@author: AnaMaria
"""
import pandas as pd
import time
from TwitterAPI import TwitterAPI
from datetime import datetime
import re 
import urllib.request, os, csv

consumer_key1 = '4HraQFYYxwubnGWiAMbnDttby'
consumer_secret1 = '5pH3aCAR25UBh9DmafYMLZrndc02viNFmkun0Aav7iIb1E5O1M'
access_token1 = '904353718036037632-rd73gTg2VGJcEuAziK7XTrRRHqrmB3N'
access_token_secret1 = 'ecMOYInArrNOW87wmAfDrzGcahVZdtYZcfnIOPJ1VmQaC'

consumer_key2 = 't320Ul55zPtvr9fmOMZMAXnOK'
consumer_secret2 = 'mGNm6KQlttbwv7vAnRsdhlrZ1IP09UJ502t7oMD8SjwF3HCjw3'
access_token2 = '930204361942360064-ApgGNXpNK1wGfCc2clk3B0gGnnuxE5f'
access_token_secret2 = 'Ad5gFxLIl3IFV63RHyuTLSZk9CPdaWygOLDPFbLx4tkwM'

consumer_key3 = 'yWnEGpPZrN6XWkbrV51uw1Eev'
consumer_secret3 = '7uVwDIP1OmMQKinUW9NVMyjVFJZTj8Jn7nrAnBf3apR5m8Odt3'
access_token3 = '930215531571007494-NBdxl58tcCWkMQdBc77P3tvEeQQmeO8'
access_token_secret3 = 'wikdEUEASu9lrmTRUuOANJdpfN04gYZlK51rG8GCbKlyS'

consumer_key4 = '4r7Er8kiXh8RqcSJb8GA4vHp0'
consumer_secret4 = '	l6XEwAeUdr8eOmE92VFUEzMMjh56VwGoSnKfhMT8Iot9pufDjP'
access_token4 = '4r7Er8kiXh8RqcSJb8GA4vHp0'
access_token_secret4 = '	l6XEwAeUdr8eOmE92VFUEzMMjh56VwGoSnKfhMT8Iot9pufDjP'

filename = 'company_update.txt'
filename2 = 'company_location.txt'
filename1 = 'cluster_location.txt'
def get_twitter(number):
    if number ==1: 
        return TwitterAPI(consumer_key1, consumer_secret1, access_token1, access_token_secret1)
    elif number == 2: 
        return TwitterAPI(consumer_key2, consumer_secret2, access_token2, access_token_secret2)
    elif number ==3: 
        return TwitterAPI(consumer_key3, consumer_secret3, access_token3, access_token_secret3)
    elif number == 4: 
        return TwitterAPI(consumer_key4, consumer_secret4, access_token4, access_token_secret4)
    

def robust_request(resource, params, max_tries=1000):
    flag = 0 
    for i in range(max_tries):
        for i in range(1,4):
            twitter = get_twitter(i)
            request = twitter.request(resource, params)
            if request.status_code == 200:
                flag = 1
                return request                
                break
        if flag == 0:
            print('Sleeping. ')
            time.sleep(900)
    
    
def fetchGF(googleticker):
    url="http://www.google.com/finance?&q=" + googleticker
    txt=urllib.request.urlopen(url).read()
    k=re.search('id="ref_(.*?)">(.*?)<',str(txt))
    if k:
        tmp=k.group(2)
        q=tmp.replace(',','')
    else:
        q="Nothing found for: "+googleticker
    return q


def combine(ticker, number):
    quote=fetchGF(ticker) 
    t=datetime.utcnow()
    output=[number, t.year,t.month,t.day,t.hour,t.minute, t.second,quote]
    return output


def stock_value():
    tickers=["INDEXSP:.INX"]

    i = 1
    freq=60
    fname="stock_111.csv"
    os.path.exists(fname) and os.remove(fname)
    
    f = open(fname,'w')
    da = ['Year','Month', 'Day', 'Hour', 'Minute', 'Sec', 'Quote' ]
    writer=csv.writer(f, lineterminator='\n',)
    with f:
        writer.writerow(da)
        while True:
            f = open(fname,'w', newline='')
            data=combine(tickers[0], i)
            i=i+1
            writer.writerow(data)
            f.close()
            time.sleep(freq)
        f.close()
        
        
def get_stock_value(time):
    time = re.search( r'(.*?) (.*?) (.*?) (.*?):(.*?):(.*?) (.*?)$', time, re.M|re.I|re.U)
    name = 'stock_value.csv'
    time_tweet = [int(time.group(4))+6, int(time.group(5)),int(time.group(6))]
    if time_tweet[0]> 19:
        time_tweet = [13, 30, 00]
    stock = pd.read_csv(name)
    time_fin = int(time_tweet[0])*3600 + int(time_tweet[1])*60 + int(time_tweet[2])
    time_fin1 = [time_fin//3600, (time_fin%3600)//60, (time_fin%60)]
    time_final1 = int(time_tweet[0])*3600 + int(time_tweet[1])*60 + int(time_tweet[2]) + 10*60
    time_final = [time_final1//3600, (time_final1%3600)//60, (time_final1%60)]
    stock_value1 = stock.loc[(stock['Hour'] == time_fin1[0]) & (stock['Minute'] == time_fin1[1])]['Quote'].values
    len(stock_value1)
    stock_value = stock.loc[(stock['Hour'] == time_final[0]) & (stock['Minute'] == time_final[1])]['Quote'].values
    return stock_value, time_tweet , stock_value1       

def tokenize(tweet):
    tokens = []
    tweet = tweet.lower()
    tweet = re.sub('@\S+', ' ', tweet)
    tweet = re.sub('http\S+', ' ', tweet)
    tweet = re.sub('[!]', ' !', tweet)
    tweet = re.sub('[?]', ' ?', tweet)
    tweet = re.sub('[!]', 'exclamationpoint', tweet)
    tweet = re.sub('[?]', 'questionmark', tweet)
    tokens.append(re.findall('[A-Za-z]+', tweet))
    return tokens


def get_tweets(screen_names):
    filedata = open("test.csv",'w')
    da = ['Tweet','Time_tweet']
    writer=csv.writer(filedata, lineterminator='\n')
    writer.writerow(da)
    number = 0 
    for screen, idif in screen_names:
        timeline = [tweet for tweet in robust_request("statuses/user_timeline", {'screen_name': screen, 'since_id':idif, 'count':10},5)]        
        for tweet in timeline:
            stock_val, time_tweet, stock_value1 = get_stock_value(tweet['created_at'])
            number +=1
            tweet_final = tokenize(tweet['text'])
            data = [tweet_final, time_tweet]      
            writer.writerow(data)
    return number 

def get_friends(screen_names):
    f = open(filename1, "w", encoding="utf-8")
    lista1= []
    for idif in screen_names:
        timeline = [tweet for tweet in robust_request("friends/list", {'screen_name':idif, 'count':10},5)]
        if timeline != 0: 
            for friend in timeline: 
                lista1.append((friend['name'],friend['location']))
            f.write("%s\n" %lista1)
            lista1 = []
    f.close()
def read_sinceidtxt(filename, number):
    lista = []
    lista1 = []
    file = open(filename, "r")
    candidates = file.read().splitlines()
    for can in candidates:
        if number == 0:
            can = re.sub('[-]', ' ', can)
            can = can.split()       
            lista.append((can[0], int(can[1])))
        else: 
            can = can.split('\', \'')
            can = can[0].split('\", \'')
            can = re.sub('[\'|(|\[]', '', can[0])
            lista1.append(can)
    return lista, lista1

def update_read_sinceidtxt(lista):
    f = open("company_update.txt","w")
    for screen, timeline in lista.items(): 
        f.write("%s - %d\n" %(screen, timeline))
    f.close()

def main():
    sinceid, nothing = read_sinceidtxt(filename, 0)
    nothing, candidates = read_sinceidtxt(filename2, 1)
    number = get_tweets(sinceid)
    get_friends(candidates)

    return len(sinceid), number
    
    
if __name__ == '__main__':
    main()