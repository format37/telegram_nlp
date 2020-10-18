# Sentiment analysis of Telegram messages (Russian model)
# https://demo.deeppavlov.ai/#/ru/sentiment

from deeppavlov import build_model, configs
import json
import pandas as pd
import os
import datetime

BATCH_SIZE = 1200 # reduce if memory overflow
os.environ["CUDA_VISIBLE_DEVICES"]="0"
input_file_name = 'in.json'
output_file_name = 'out.csv'

def solve_sentences(df):

	sentences = df.text

	model = build_model(configs.classifiers.rusentiment_bert, download=False) #download first time
	print('sentences',sentences.shape)
	res = model(sentences)

	df['res'] = res

	return df

def report(df_res):

	positive = df_res[df_res.res=='positive']
	neutral  = df_res[df_res.res=='neutral']
	negative = df_res[df_res.res=='negative']
	speech = df_res[df_res.res=='speech']
	skip = df_res[df_res.res=='skip']

	res_z = neutral[{'from','res'}].groupby('from').res.count()
	res_n = negative[{'from','res'}].groupby('from').res.count()
	res_p = positive[{'from','res'}].groupby('from').res.count()
	res_h = speech[{'from','res'}].groupby('from').res.count()
	res_k = skip[{'from','res'}].groupby('from').res.count()

	final_z = pd.DataFrame({'from':res_z.index, 'negative':res_z.values})
	final_n = pd.DataFrame({'from':res_n.index, 'neutral':res_n.values})
	final_p = pd.DataFrame({'from':res_p.index, 'positive':res_p.values})
	final_h = pd.DataFrame({'from':res_h.index, 'speech':res_h.values})
	final_k = pd.DataFrame({'from':res_k.index, 'skip':res_k.values})

	df_final = pd.DataFrame(columns=['from', 'positive', 'neutral', 'negative','speech','skip'])
	df_final = df_final.append(final_z)
	df_final = df_final.append(final_n)
	df_final = df_final.append(final_p)
	df_final = df_final.append(final_h)
	df_final = df_final.append(final_k)

	df_final = df_final.fillna(0)

	df_group = df_final.groupby('from').sum()

	df_group['score']=(df_group.positive+df_group.speech+df_group.skip+df_group.neutral-df_group.negative)/(df_group.positive+df_group.positive+df_group.speech+df_group.skip+df_group.negative+df_group.neutral)
	df_group.sort_values('score', ascending = True)
	df_group['from'] = df_group.index
	
	df_group = df_group.sort_values('score', ascending = False)
	
	df_group.to_csv('report.csv')

	graphic = df_group.plot(
		x='from',
		y='score',
		kind='barh',
		title = 'sentiment',
		figsize=(15,15),
		grid = True
	)
	fig = graphic.get_figure()
	fig.savefig("report.png")

# main	

with open(input_file_name) as f:
	file_data = json.load(f)

print('json --> df..')
batch = 0
row=0
print('batch',batch)
df = pd.DataFrame(columns=['batch','date','month','from','text'])
for msg in file_data['messages']:    
	if type(msg['text'])==type('') and msg['text']!='':
		# {'id': 1126946, 'type': 'message', 'date': '2020-10-14T00:24:04', 'from': 'Alex j', 'from_id': 4448833589, 'text': 'привет'}
		date = datetime.datetime.strptime(msg['date'],"%Y-%m-%dT%H:%M:%S")
		df = df.append({'batch':batch, 'date':date, 'month':date.month, 'from':msg['from'], 'text':msg['text']}, ignore_index=True)

		row+=1
		if row>BATCH_SIZE:
			batch+=1
			row=0
			print('batch',batch)
	else:
		# example 1
		'''
		{'id': 1127162, 'type': 'message', 'date': '2020-10-14T12:07:30', 'from': 'Alex bg', 'from_id': 4348007839, 'forwarded_from': 'Bananasoup', 'file': '(File not included. Change data exporting settings to download.)', 'thumbnail': '(File not included. Change data exporting settings to download.)', 'media_type': 'video_file', 'mime_type': 'video/mp4', 'duration_seconds': 15, 'width': 448, 'height': 624, 'text': ''}
		'''
		# example 2
		'''
		{'id': 920016, 'type': 'message', 'date': '2019-11-13T14:19:55', 'from': 'Vlad k', 'from_id': 4420223003, 'text': [{'type': 'link', 'text': 'https://news.mail.ru'}]}
		'''
		# example 3
		'''{'id': 920018, 'type': 'message', 'date': '2019-11-13T14:20:45', 'from': 'Vlad k', 'from_id': 4420223003, 'photo': '(File not included. Change data exporting settings to download.)', 'width': 1173, 'height': 140, 'text': ''}
		'''
		# example 4
		'''
		{'id': 920044, 'type': 'service', 'date': '2019-11-13T20:58:12', 'actor': 'Alex m', 'actor_id': 4365571815, 'action': 'invite_members', 'members': ['Alex j'], 'text': ''}
		'''
		# example 5
		'''
		{'id': 920523, 'type': 'message', 'date': '2019-11-13T22:37:06', 'from': 'Alex m', 'from_id': 4365571815, 'file': '(File not included. Change data exporting settings to download.)', 'thumbnail': 'stickers/sticker.webp_thumb.jpg', 'media_type': 'sticker', 'sticker_emoji': '😖', 'width': 512, 'height': 400, 'text': ''}
		'''
		# example 6
		'''
		{'id': 920971, 'type': 'message', 'date': '2019-11-14T12:01:36', 'from': 'Alex l', 'from_id': 4432196399, 'text': [{'type': 'mention', 'text': '@format37'}, ' она учится?']}
		'''		
		pass

print('solving..')
df_res = pd.DataFrame(columns=['batch','month','from','text','res'])
for d in df.batch.unique():	
	print('batch',d)
	df_res = df_res.append( solve_sentences(df[df.batch==int(d)]) )

print('save sentiment data to csv..')
df_res.to_csv(output_file_name)

print('report..')
report(df_res)

print('Happy end! exit..')