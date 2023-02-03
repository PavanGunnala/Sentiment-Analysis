from flask import Flask, render_template
import pandas as pd
import json
import plotly
import plotly.express as px

app = Flask(__name__)

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
# import plotly.express as px


@app.route('/')
def notdash():
   # df = pd.DataFrame({
   #    'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 'Bananas'],
   #    'Amount': [4, 1, 2, 2, 4, 5],
   #    'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
   # })
   df = pd.read_json('System.hpv_US_tweets.json', lines=True)
   df.drop(columns=['id'], inplace=True)
   df = df.drop_duplicates('text')

   def clean_tweet_text(text):
      text = re.sub(r'@\w+', '', text)
      text = re.sub(r'#', '', text)
      text = re.sub(r'RT[\s]+', '', text)
      text = re.sub(r'https?:\/\/\S+', '', text)
      text = text.lower()
      return text


# the following line makes use of an apply function-- it will call clean_tweet_text on every element in the 'text' column
   df['text'].transform(clean_tweet_text)
   df['created_at'] = pd.to_datetime(df['created_at']).dt.date
   df[df['truncated'] == True].head()
   df[df['truncated'] == True].head()
   pd.set_option('display.max_colwidth', 400)
   df.sort_values(by='created_at', ascending=False)[['text', 'created_at', 'user', 'place', 'extended_tweet', 'favorited', 'retweeted']]
   df.sort_values(by=['created_at', 'favorited'], ascending=[True, False])[['text', 'created_at', 'user', 'place', 'extended_tweet', 'favorited', 'retweeted']]
   testimonial = TextBlob("So excited to get my vaccine!")
   print(testimonial.sentiment)
   testimonial = TextBlob("Is the vaccine painful?")
   print(testimonial.sentiment)
   testimonial = TextBlob("It's important that boys aged 12 or 13 get the #HPV vaccine.")
   print(testimonial.sentiment)
   testimonial = TextBlob("Most sexually active persons will, at some point, be exposed to #HPV")
   print(testimonial.sentiment)
   text = """
    #HPV is a common virus that can lead to six types of cancers later in life. 

    There is no cure but it can be prevente… https://t.co/wu5CQtKmaF
   """
   blob = TextBlob(text)
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('brown')
   print("Tags")
   print(blob.tags, end="\n\n")

   print("Noun Phrases")
   print(blob.noun_phrases, end="\n\n")

   print("Words")
   print(blob.words, end="\n\n")

   print("Sentences")
   print(blob.sentences, end="\n\n")
   for sentence in blob.sentences:
      print(sentence)
      print("polarity:", sentence.sentiment.polarity)
      print("subjectivity:", sentence.sentiment.subjectivity)
      print()
   df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
   df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
   fig3 = px.histogram(df['polarity'])

   fig3graphJSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
   fig4 = px.histogram(df['subjectivity'])
   fig4graphJSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
   # what are the top 10 most retweeted tweets
   df.sort_values(by='polarity', ascending=True)[['text', 'polarity', 'subjectivity']].reset_index(drop=True)
   df.sort_values(by='polarity', ascending=False)[['text', 'polarity', 'subjectivity']].reset_index(drop=True)
   df.sort_values(by='subjectivity', ascending=True)[['text', 'polarity', 'subjectivity']].reset_index(drop=True)
   df.sort_values(by='subjectivity', ascending=False)[['text', 'polarity', 'subjectivity']].reset_index(drop=True)
   timeline = df.groupby(['created_at']).count().reset_index()
   timeline['count'] = timeline['text']
   timeline = timeline[['created_at', 'count']]
   fig = px.bar(timeline, x='created_at', y='count', labels={'date': 'Date', 'count': 'Tweet Count'})
   # what are the top 10 most retweeted tweets
   # x = df['source'].value_counts().head(n=10)
   # fig = px.bar(x = df['source'].value_counts().head(n=10),y = 10)
   # fig = px.bar(df[''], x='Fruit', y='Amount', color='City',    barmode='group')
   graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
   criteria = [df['polarity'].between(-1, -0.01), df['polarity'].between(-0.01, 0.01), df['polarity'].between(0.01, 1)]
   values = ['negative', 'neutral', 'positive']
   df['sentiment'] = np.select(criteria, values, 0)
   fig2  = px.bar(df['sentiment'].value_counts().sort_index())
   # plot sentiment counts
   fig2graphJSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
   timeline = df.groupby(['created_at']).agg(np.nanmean).reset_index()
   timeline['count'] = df.groupby(['created_at']).count().reset_index()['retweeted']
   timeline = timeline[['created_at', 'count', 'polarity', 'retweeted', 'favorited', 'subjectivity']]
   timeline["polarity"] = timeline["polarity"].astype(float)
   timeline["subjectivity"] = timeline["subjectivity"].astype(float)
   # edit 2222
   
   timeline.sort_values(by='polarity', ascending=True)
   
   
   df[df['created_at'].astype(str) < '2021-01-12'][['place', 'created_at', 'text', 'polarity', 'subjectivity', 'sentiment']]
   fig5 = px.bar(timeline, x='created_at', y='count', color='polarity')
   fig5graphJSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)

   fig6 = px.bar(timeline, x='created_at', y='count', color='subjectivity')
   fig6graphJSON = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)
   return render_template('notdash.html',fig4graphJSON =fig4graphJSON,fig3graphJSON=fig3graphJSON, graphJSON=graphJSON,fig2graphJSON=fig2graphJSON,fig5graphJSON=fig5graphJSON,fig6graphJSON=fig6graphJSON)