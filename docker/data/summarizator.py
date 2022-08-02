import pandas as pd
from transformers import pipeline
from datetime import datetime as dt

#pipeline = pipeline(TASK, 
#                    model=MODEL_PATH,
#                    device=1,     # to utilize GPU cuda:1
#                    device=0,     # to utilize GPU cuda:0
#                    device=-1)    # default value which utilize CPU

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

# df = pd.read_csv('cc.csv')
df = pd.read_csv('summarized.csv')

# crop text to 1000 if it's longer than 1000
df['text'] = df['text'].apply(lambda x: x[:1000] if len(x) > 1000 else x)

texts = df['text']

# convert to list
texts = texts.to_list()

# summarize all texts
print(dt.now(), 'start', len(df))
summarized_texts = summarizer(texts, min_length=10, max_length=30)
print(dt.now(), 'end')

# Add sumary_text to df
for i, text in enumerate(summarized_texts):
    df.loc[i, 'summary_text30'] = text['summary_text']

# save to csv
# df.to_csv('summarized.csv')
df.to_csv('summarized2.csv')

print(dt.now(), 'done')
