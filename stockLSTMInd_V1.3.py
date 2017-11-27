from __future__ import print_function

import os
import sys
import keras
import nltk
import xlrd
import datetime
import math
import numpy as np
from textblob import TextBlob
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten, Dropout, Activation, Concatenate, LSTM, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D, Embedding, Reshape, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from keras.callbacks import LearningRateScheduler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#INPUT:
# 1. Twitter Data Preprocessed file for a company.
# 2. Individually Standardized Stock Indicators for that company.
# 3. Raw Stock Indicators for that company.


BASE_DIR = ''
GLOVE_DIR = BASE_DIR + 'glove.6B/'
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.3

#Building index mapping words in the set
#to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#Prepare text samples and their labels
print('Processing text dataset')

tweetBookI = xlrd.open_workbook("Microsofti.xlsx")
stockBookO = xlrd.open_workbook("MSFTo.xlsx")
rawbookO = xlrd.open_workbook("MSFT.xlsx")

datei = []
dateo = []
texts = []
temp_texts = []
sentences = []
encoded_data = []
labels = []
features = []
temp_features = []
out = []
temp = []
last_date = "1900-01-01"
count=0

tweetSheetI = tweetBookI.sheet_by_index(0)
stockSheetO = stockBookO.sheet_by_index(0)
rawSheetO = rawbookO.sheet_by_index(0)

#Reading Raw Stock Indicators for de-standardizing:
for row in range(1, rawSheetO.nrows):
    #The output value to be predicted by the model:
    #1:Open 2:High 3:Low 4:Close 5:Adj Close 6:Volume
    out.append(float(rawSheetO.cell_value(row,1)))

#Creating StandardScaler Object to de-standardize the outputs later:
out = (np.asarray(out)).reshape((len(out), 1))
scaler_out = StandardScaler().fit(out)

prev_date = '2017-03-01'
print('Processing Tweet Data....')
for row in range(0, tweetSheetI.nrows):
	cur_date = str(datetime.datetime(*xlrd.xldate_as_tuple(tweetSheetI.cell_value(row,0), tweetBookI.datemode)).date())
	if cur_date != prev_date:
		if len(temp_texts) < 2000:
			temp_texts.extend([0] * (2000 - len(temp_texts)))
		elif len(temp_texts) > 2000:
			temp_texts = temp_texts[0:2000]
		texts.append(temp_texts)
		if len(temp_features) < 2000:
			temp_features.extend([[0.0,0.0,0.0,0.0,0.0,0.0,0.0]] * (2000 - len(temp_features)))
		elif len(temp_features) > 2000:
			temp_features = temp_features[0:2000]
		features.append(temp_features)
		datei.append(prev_date)
		temp_texts = []
		temp_features = []
		for row1 in range(1, stockSheetO.nrows):
			last_date = str(datetime.datetime(*xlrd.xldate_as_tuple(stockSheetO.cell_value(row1,0), stockBookO.datemode)).date())
			if prev_date == last_date:
				temp = []
				#The output value to be predicted by the model:
				#1:Open 2:High 3:Low 4:Close 5:Adj Close 6:Volume
				temp.append(float(stockSheetO.cell_value(row1,1)))
				#temp.append(float(stockSheetO.cell_value(row1,2)))
				#temp.append(float(stockSheetO.cell_value(row1,3)))
				#temp.append(float(stockSheetO.cell_value(row1,4)))
				#temp.append(float(stockSheetO.cell_value(row1,5)))
				#temp.append(float(stockSheetO.cell_value(row1,6)))
				labels.append(temp)
				break
		prev_date = cur_date

	temp1 = []
	#Tweet text:	
	tweet = str(tweetSheetI.cell_value(row,7))
	temp_texts.append(tweet)

	#Number of friendCount:
	temp1.append(float(tweetSheetI.cell_value(row,2)))

	#Number of followerCount:
	temp1.append(float(tweetSheetI.cell_value(row,3)))

	#Number of retweets:
	temp1.append(float(tweetSheetI.cell_value(row,4)))

	#Number of favorites:
	temp1.append(float(tweetSheetI.cell_value(row,5)))

	#Whether tweet has hyperlink or not:
	if str(tweetSheetI.cell_value(row,8)) == "yes":
		temp1.append(1)
	else:
		temp1.append(0)

	#Getting Polarity and Subjectivity of the tweets using TextBlob
	sentiments = TextBlob(tweet)
	temp1.append(sentiments.sentiment.polarity)
	temp1.append(sentiments.sentiment.subjectivity)
	temp_features.append(temp1)

	if row == tweetSheetI.nrows - 1:
		if len(temp_texts) < 2000:
			temp_texts.extend([0] * (2000 - len(temp_texts)))
		elif len(temp_texts) > 2000:
			temp_texts = temp_texts[0:2000]
		texts.append(temp_texts)
		if len(temp_features) < 2000:
			temp_features.extend([[0.0,0.0,0.0,0.0,0.0,0.0,0.0]] * (2000 - len(temp_features)))
		elif len(temp_features) > 2000:
			temp_features = temp_features[0:2000]
		features.append(temp_features)
		datei.append(prev_date)
		for row1 in range(1, stockSheetO.nrows):
			last_date = str(datetime.datetime(*xlrd.xldate_as_tuple(stockSheetO.cell_value(row1,0), stockBookO.datemode)).date())
			if prev_date == last_date:
				temp = []
				#The output value to be predicted by the model:
				#1:Open 2:High 3:Low 4:Close 5:Adj Close 6:Volume
				temp.append(float(stockSheetO.cell_value(row1,1)))
				#temp.append(float(stockSheetO.cell_value(row1,2)))
				#temp.append(float(stockSheetO.cell_value(row1,3)))
				#temp.append(float(stockSheetO.cell_value(row1,4)))
				#temp.append(float(stockSheetO.cell_value(row1,5)))
				#temp.append(float(stockSheetO.cell_value(row1,6)))
				labels.append(temp)
				break


row_count = len(texts)
print('Found %s texts.' % row_count)


texts = np.asarray(texts)
sentences = [item for sublist in texts for item in sublist]
sentences = np.asarray(sentences)

#Vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#Bag of Words:
print('Apply Porter Stemmer')
porter_stemmer = nltk.stem.PorterStemmer()
for i,tweet in enumerate(sentences):
	words = nltk.word_tokenize(tweet)
	for j,word in enumerate(words):
		words[j] = porter_stemmer.stem(word)
	sentences[i] = ' '.join(words)

print('Converting text to Bag of Words')

cv = CountVectorizer()
encoded_data = cv.fit_transform(sentences).toarray()

print('Bag of Words processing complete')
sequences =np.asarray(sequences)
encoded_data =np.asarray(encoded_data)

#data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
data = np.asarray(sequences)
data = np.reshape(data, (row_count, 2000))
encoded_data = np.reshape(encoded_data, (row_count, 2000, encoded_data.shape[1]))

data_new = []
for l in data:
	flat_list = []
	for sublist in l:
		for item in sublist:
			flat_list.append(item)	
	data_new.append(flat_list)

data = np.asarray(data_new)
data = pad_sequences(data)
input1_shape = data.shape[1]

labels = np.asarray(labels)
features = np.asarray(features)
print('Shape of data tensor:', data.shape)
print('Shape of Encoded data tensor:', encoded_data.shape)
print('Shape of features tensor:', features.shape)
print('Shape of label tensor:', labels.shape)



#Prepare embedding matrix
print('Preparing embedding matrix.')
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words+1, EMBEDDING_DIM))
for word, i in enumerate(word_index.items()):
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
	embedding_matrix[i-1] = embedding_vector

print('Embedding Done')

#Separate Train and Test Data:
print('Separating test and train data')
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
#datei = datei[indices]
encoded_data = encoded_data[indices]
labels = labels[indices]
features = features[indices]

indexval = int(0.6 * data.shape[0])
indextest = int(0.8 * data.shape[0])

#x_train_date = datei[:indexval]
x_train_data = data[:indexval]
x_train_edata = encoded_data[:indexval]
x_train_features = features[:indexval]
y_train = labels[:indexval]

#x_val_date = datei[indexval:indextest]
x_val_data = data[indexval:indextest]
x_val_edata = encoded_data[indexval:indextest]
x_val_features = features[indexval:indextest]
y_val = labels[indexval:indextest]

#x_test_date = datei[indextest:]
x_test_data = data[indextest:]
x_test_edata = encoded_data[indextest:]
x_test_features = features[indextest:]
y_test = labels[indextest:]


print('Shape of x_train_data tensor:', x_train_data.shape)
print('Shape of x_train_edata tensor:', x_train_edata.shape)
print('Shape of x_train_features tensor:', x_train_features.shape)
print('Shape of y_train tensor:', y_train.shape)
print('Shape of x_val_data tensor:', x_val_data.shape)
print('Shape of x_val_edata tensor:', x_val_edata.shape)
print('Shape of x_val_features tensor:', x_val_features.shape)
print('Shape of y_val tensor:', y_val.shape)
print('Shape of x_test_data tensor:', x_val_data.shape)
print('Shape of x_test_edata tensor:', x_val_edata.shape)
print('Shape of x_test_features tensor:', x_val_features.shape)
print('Shape of y_test tensor:', y_val.shape)



input2_shape = x_train_edata.shape[2]

print('Training model.')

############# Model 1: 1D CNN with embedding matrix

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
# note that we set trainable = False so as to keep the embeddings fixed

embedding_layer = Embedding(num_words+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=input1_shape,
                            trainable=False)

#Input Layer
input1 = Input(shape=(input1_shape, ), dtype='float32')
embedded_sequences = embedding_layer(input1)
	
#Add Convolution1D layers, which will learn filters
x1 = Conv1D(128, 3, padding='same', activation='relu', strides=1)(embedded_sequences)
x1 = MaxPooling1D(pool_size=3, strides=1, padding='same')(x1)

x1 = Conv1D(128, 3, padding='same', activation='relu', strides=1)(x1)
x1 = MaxPooling1D(pool_size=3, strides=1, padding='same')(x1)

x1 = Conv1D(128, 3, padding='same', activation='relu', strides=1)(x1)
x1 = MaxPooling1D(pool_size=3, strides=1, padding='same')(x1)

x1 = Flatten()(x1)

#Add a vanilla hidden layer:
x1 = Dense(128, activation='linear')(x1)
x1 = Dropout(0.2)(x1)
x1 = Activation('linear')(x1)

#Project onto a single output layer:
x1 = Dense(10, activation='linear')(x1)


############# Model 2: 1D CNN with Bag-of-Words

#Input Layer
input2 = Input(shape=(2000, input2_shape), dtype='float32')

#Add Convolution1D layers, which will learn filters
x2 = Conv1D(128, 3, padding='same', activation='relu', strides=1)(input2)
x2 = MaxPooling1D(pool_size=3, strides=1, padding='same')(x2)

x2 = Conv1D(128, 3, padding='same', activation='relu', strides=1)(x2)
x2 = MaxPooling1D(pool_size=3, strides=1, padding='same')(x2)

x2 = Conv1D(128, 3, padding='same', activation='relu', strides=1)(x2)
x2 = MaxPooling1D(pool_size=3, strides=1, padding='same')(x2)

x2 = Flatten()(x2)

#Add a vanilla hidden layer:
x2 = Dense(128, activation='linear')(x2)
x2 = Dropout(0.2)(x2)
x2 = Activation('linear')(x2)

#Project onto a single output layer:
x2 = Dense(10, activation='linear')(x2)

############ Model 3: 1D CNN for other features

#Input Layer
input3 = Input(shape=(2000,7), dtype='float32')

#Add Convolution1D layers, which will learn filters
x3 = Conv1D(128, 2, padding='same', activation='relu', strides=1)(input3)
x3 = Conv1D(128, 2, padding='same', activation='relu', strides=1)(x3)
x3 = Conv1D(128, 2, padding='same', activation='relu', strides=1)(x3)

x3 = Flatten()(x3)

#Add a vanilla hidden layer:
x3 = Dense(128, activation='linear')(x3)
x3 = Dropout(0.2)(x3)
x3 = Activation('linear')(x3)

#Project onto a single output layer:
x3 = Dense(10, activation='linear')(x3)

#We now concatenate the three CNNs:
x4 = keras.layers.concatenate([x1, x2, x3])

#Add three fully-connected layers:
x4 = Reshape((1,30))(x4)
x4 = LSTM(128)(x4)
x4 = Dense(128, activation='linear')(x4)
x4 = Dense(128, activation='linear')(x4)
x4 = Dense(128, activation='linear')(x4)

#Final Output layer:
x4 = Dense(1, activation='linear')(x4)

model = Model(inputs=[input1, input2, input3], outputs=[x4])

#Compile Model
model.compile(loss='mean_absolute_error', optimizer='rmsprop')

model.summary()
model.fit([x_train_data, x_train_edata, x_train_features], [y_train], batch_size=500, epochs=200,
           validation_data=([x_val_data, x_val_edata, x_val_features], [y_val]))

score = model.evaluate([x_val_data, x_val_edata, x_val_features], [y_val], batch_size=3000)
print('Final Loss for Validation Data: ', score)

#Saving Model:
model.save('modelIndMSFT.h5')

#Predicting the Stock Values with trained model:
predictions = model.predict([x_test_data, x_test_edata, x_test_features])
actual = []
pred = []
for i, (pr,ac) in enumerate(zip(predictions,y_test)):
    actual.append(scaler_out.inverse_transform(ac))
    pred.append(scaler_out.inverse_transform(pr))

pred = np.asarray(pred)
actual = np.asarray(actual)

#Plotting the Actual and predicted values:
PredictPlot = np.empty_like(pred)
PredictPlot[:, :] = np.nan
PredictPlot[0:len(pred)+1, :] = pred

plt.plot(actual,'g')
plt.plot(PredictPlot,'r')
plt.show()

