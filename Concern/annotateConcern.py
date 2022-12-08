import pandas as pd
import spacy
import re
from string import digits
import time
import os

# try:
#     os.system("python3 -m spacy download fr_core_news_md")
# except:
#     os.system("python -m spacy download fr_core_news_md")

nlp = spacy.load('fr_core_news_md',disable=["parser", "ner"])
irrelevant_chars="~?!./\:;+=&^%$#@(,)"
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
remove_digits = str.maketrans('', '', digits)
remove_chars = str.maketrans('', '', irrelevant_chars)

''' Function to clean tweets. Performs lowercasing, replacing irrelevant characters, numbers and special symbols'''
def clean(x):
    x=x.lower()
    x=x.replace('\xa0','')
    x=x.replace('rt ','')
    x=x.replace('#','')
    x=" ".join(filter(lambda a:a[0]!='@', x.split()))
    x=re.sub(r'http\S+', '', x)
    x = x.translate(remove_digits)
    x = x.translate(remove_chars)
    x = emoji_pattern.sub(r'', x)
    x=x.replace('!','')
    x=x.replace('?','')
    x=x.replace('@','')
    x=x.replace('&','')
    x=x.replace('$','')
    x=x.replace('\x92',' ')
    x=[t for t in x.split() if len(t)>0]
    x=' '.join(x)
    return x

''' Lemmatizes words to obtain root form '''
def lemmatize(sent):
    s=[token.lemma_ for token in nlp(sent)]
    s=' '.join(s)
    return s

''' Function to generate annotation object that will be returned '''
def generate_annotation_object(tweetid,text,annot_type,annot):
    result={
        "id":tweetid,
        "type":annot_type,
        "text":text,
        "confidence":annot,
        "providerName":"ta1-usc-isi"
    }
    return result

''' Function to score 3.1 Ukraine concern '''
def scoreUkraine(tweet_object):

    #Loads keywords from secondary memory. Change path as necessary
    ukraine_lexicon=pd.read_csv('./lexicon/fr_all/ukraine_fr_sage.csv')['words'].tolist()

    id=tweet_object['id']
    tweet_text=tweet_object['contentText']
    cleaned_text=clean(tweet_text)
    lemmatized_text=lemmatize(cleaned_text)
    tweet_text_terms=lemmatized_text.split()
    search_bool=[True if t in ukraine_lexicon else False for t in tweet_text_terms]
    if True in search_bool:
        return generate_annotation_object(id,"Ukraine","concern-3.1",1)
    return generate_annotation_object(id,"Ukraine","concern-3.1",0)

''' Function to score 3.2 Russia concern '''
def scoreRussia(tweet_object):

    #Loads keywords from secondary memory. Change path as necessary
    russia_lexicon=pd.read_csv('./lexicon/fr_all/russia_fr_sage.csv')['words'].tolist()

    id=tweet_object['id']
    tweet_text=tweet_object['contentText']
    cleaned_text=clean(tweet_text)
    lemmatized_text=lemmatize(cleaned_text)
    tweet_text_terms=lemmatized_text.split()
    search_bool=[True if t in russia_lexicon else False for t in tweet_text_terms]
    if True in search_bool:
        return generate_annotation_object(id,"Russia","concern-3.2",1)
    return generate_annotation_object(id,"Russia","concern-3.2",0)

''' Function to score 3.3 NATO concern '''
def scoreNATO(tweet_object):

    #Loads keywords from secondary memory. Change path as necessary
    NATO_lexicon=pd.read_csv('./lexicon/fr_all/NATO_fr_sage.csv')['words'].tolist()

    id=tweet_object['id']
    tweet_text=tweet_object['contentText']
    cleaned_text=clean(tweet_text)
    lemmatized_text=lemmatize(cleaned_text)
    tweet_text_terms=lemmatized_text.split()
    search_bool=[True if t in NATO_lexicon else False for t in tweet_text_terms]
    if True in search_bool:
        return generate_annotation_object(id,"NATO","concern-3.3",1)
    return generate_annotation_object(id,"NATO","concern-3.3",0)

''' Function to score 3.4 Refugees concern '''
def scoreRefugees(tweet_object):

    #Loads keywords from secondary memory. Change path as necessary
    refugees_lexicon=pd.read_csv('./lexicon/fr_all/refugees_fr_sage.csv')['words'].tolist()

    id=tweet_object['id']
    tweet_text=tweet_object['contentText']
    cleaned_text=clean(tweet_text)
    lemmatized_text=lemmatize(cleaned_text)
    tweet_text_terms=lemmatized_text.split()
    search_bool=[True if t in refugees_lexicon else False for t in tweet_text_terms]
    if True in search_bool:
        return generate_annotation_object(id,"Refugees","concern-3.4",1)
    return generate_annotation_object(id,"Refugees","concern-3.4",0)

''' Function to score 3.5 Defense concern '''
def scoreDefense(tweet_object):

    #Loads keywords from secondary memory. Change path as necessary
    defense_lexicon=pd.read_csv('./lexicon/fr_all/defense_fr_sage.csv')['words'].tolist()

    id=tweet_object['id']
    tweet_text=tweet_object['contentText']
    cleaned_text=clean(tweet_text)
    lemmatized_text=lemmatize(cleaned_text)
    tweet_text_terms=lemmatized_text.split()
    search_bool=[True if t in defense_lexicon else False for t in tweet_text_terms]
    if True in search_bool:
        return generate_annotation_object(id,"Defense","concern-3.5",1)
    return generate_annotation_object(id,"Defense","concern-3.5",0)

''' Function to score 3.6 Economy concern '''
def scoreEconomy(tweet_object):

    #Loads keywords from secondary memory. Change path as necessary
    economy_lexicon=pd.read_csv('./lexicon/fr_all/economy_fr_sage.csv')['words'].tolist()

    id=tweet_object['id']
    tweet_text=tweet_object['contentText']
    cleaned_text=clean(tweet_text)
    lemmatized_text=lemmatize(cleaned_text)
    tweet_text_terms=lemmatized_text.split()
    search_bool=[True if t in economy_lexicon else False for t in tweet_text_terms]
    if True in search_bool:
        return generate_annotation_object(id,"Economy","concern-3.6",1)
    return generate_annotation_object(id,"Economy","concern-3.6",0)

''' Function to score 3.7 Economic Sanctions concern '''
def scoreEconomicSanctions(tweet_object):

    #Loads keywords from secondary memory. Change path as necessary
    economic_sanctions_lexicon=pd.read_csv('./lexicon/fr_all/economic sanctions_fr_sage.csv')['words'].tolist()

    id=tweet_object['id']
    tweet_text=tweet_object['contentText']
    cleaned_text=clean(tweet_text)
    lemmatized_text=lemmatize(cleaned_text)
    tweet_text_terms=lemmatized_text.split()
    search_bool=[True if t in economic_sanctions_lexicon else False for t in tweet_text_terms]
    if True in search_bool:
        return generate_annotation_object(id,"Economic Sanctions","concern-3.7",1)
    return generate_annotation_object(id,"Economic Sanctions","concern-3.7",0)

''' Function to score 3.8 Energy concern '''
def scoreEnergy(tweet_object):

    #Loads keywords from secondary memory. Change path as necessary
    energy_lexicon=pd.read_csv('./lexicon/fr_all/energy_fr_sage.csv')['words'].tolist()

    id=tweet_object['id']
    tweet_text=tweet_object['contentText']
    cleaned_text=clean(tweet_text)
    lemmatized_text=lemmatize(cleaned_text)
    tweet_text_terms=lemmatized_text.split()
    search_bool=[True if t in energy_lexicon else False for t in tweet_text_terms]
    if True in search_bool:
        return generate_annotation_object(id,"Energy","concern-3.8",1)
    return generate_annotation_object(id,"Energy","concern-3.8",0)

''' Function to score 3.9 None/Other concern '''
def scoreNone(tweet_object, isNone):
    id = tweet_object['id']
    if isNone == 1:
        return generate_annotation_object(id, "None/Other", "concern-3.9", 1)
    else:
        return generate_annotation_object(id, "None/Other", "concern-3.9", 0)

def annotate(tweets):
    if type(tweets) is not list:
        tweets=[tweets]
    results=[]

    #start = time.time()
    for t in tweets:
        isNone = 1
        tweet_results = []
        tweet_results.append(scoreUkraine(t))
        tweet_results.append(scoreRussia(t))
        tweet_results.append(scoreNATO(t))
        tweet_results.append(scoreRefugees(t))
        tweet_results.append(scoreDefense(t))
        tweet_results.append(scoreEconomy(t))
        tweet_results.append(scoreEconomicSanctions(t))
        tweet_results.append(scoreEnergy(t))
        for result in tweet_results:
            if result["confidence"] == 1:
                isNone = 0
                break
        tweet_results.append(scoreNone(t, isNone))

    #end = time.time()
    #print(end - start)
    return results