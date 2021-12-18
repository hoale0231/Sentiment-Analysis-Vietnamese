import numpy as np
import gensim.models as word2vec
from tensorflow.keras import layers
from tensorflow import keras 
import tensorflow as tf
from keras.models import load_model
from prep import text_preprocess, loadData, randomData

class DL:
  def __init__(self, w2vModel) -> None:
    # Load word2vec model
    self.w2vModel = w2vModel
    # Load vocab
    self.word_labels = []
    for word in list(w2vModel.index_to_key):
      self.word_labels.append(word)
    
  def comment_embedding(self, comment):
    # Create zero matrix
    matrix = np.zeros((self.sequence_length, self.w2vModel.vector_size))
    
    # Split and ensure len <= sequence_length
    words = comment.split() 
    if len(words) > self.sequence_length:
      words = words[:self.sequence_length]
    lencmt = len(words)
    
    # Transform each word to vector
    for i in range(self.sequence_length):
      indexword = i % lencmt
      if(words[indexword] in self.word_labels):
          matrix[i] = self.w2vModel[words[indexword]]
 
    return matrix
  
  def transform_input(self, X, Y):
    # Transform text to matrix
    for i in range(len(X)):
      X[i] = self.comment_embedding(X[i])

    X = np.array(X)
    X = X.reshape(X.shape[0], self.sequence_length, self.w2vModel.vector_size, 1).astype('float32')

    # Transfrom label to vector 2d
    for i in range(len(Y)):
      label_ = np.zeros(2)
      label_[int(Y[i])] = 1
      Y[i] = label_
        
    return X, np.array(Y)
  
  def fit(self, X_train, Y_train, X_test, Y_test, sequence_length = 200, epochs = 10, num_filters = 128, batch_size = 30, dropout_rate = 0.5, filter_sizes = 2):
    # Transform data from text to matrix
    self.sequence_length = sequence_length
    X_train, Y_train = self.transform_input(X_train, Y_train)
    X_test, Y_test = self.transform_input(X_test, Y_test)
    
    # Define model
    self.model = keras.Sequential()
    self.model.add(layers.Convolution2D(num_filters, (filter_sizes, self.w2vModel.vector_size),
                            input_shape=(self.sequence_length, self.w2vModel.vector_size, 1), activation='relu'))
    self.model.add(layers.MaxPooling2D(pool_size=(128, 1)))
    self.model.add(layers.Dropout(dropout_rate))
    self.model.add(layers.Flatten())
    self.model.add(layers.Dense(self.w2vModel.vector_size, activation='relu'))
    self.model.add(layers.Dense(2, activation='softmax'))
    self.model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    print(self.model.summary())
    # Train model
    self.model.fit(x = X_train, y = Y_train, batch_size = batch_size, verbose=1, epochs=epochs, validation_data=(X_test, Y_test))

  def saveModel(self, filename):
    self.model.save(filename)
    
  def loadModel(self, filename, sequence_length = 200):
    self.sequence_length = sequence_length
    self.model = load_model(filename)
    
  def predict(self, sen):
    sample = text_preprocess(sen)
    maxtrix_embedding = np.expand_dims(self.comment_embedding(sample), axis=0)
    maxtrix_embedding = np.expand_dims(maxtrix_embedding, axis=3)
    result = self.model.predict(maxtrix_embedding)
    return "Positive" if np.argmax(result) == 1 else "Negative"
    
  def score(self, X_val, Y_val):
    X_val, Y_val = self.transform_input(X_val, Y_val)
    return self.model.evaluate(X_val, Y_val)

def pipeline():
  # Load data
  pos = loadData('prep_pos.txt')
  neg = loadData('prep_neg.txt')
  X_train, Y_train = randomData(pos, neg)

  pos_test = loadData('prep_pos_test.txt')
  neg_test = loadData('prep_neg_test.txt')
  X_test, Y_test = randomData(pos_test, neg_test)
  
  # Train word2vec model
  # W2Vmodel = word2vec(X_train, size=128, window=5, min_count=1, workers=4)
  # W2Vmodel.wv.save('model/word2vec.model')
  model_embedding = word2vec.KeyedVectors.load('model/word2vec.model')

  # Train CNN model
  train = DL(model_embedding)
  train.fit(X_train, Y_train, X_test, Y_test, 300, epochs=50)
  train.saveModel('model/DLmodels.h5')
  
def validate():
  # Load validate data
  pos_val = loadData('data/prep_pos_val.txt')
  neg_val = loadData('data/prep_neg_val.txt')
  X_val, Y_val = randomData(pos_val, neg_val)  

  # Load word2vec model
  model_embedding = word2vec.KeyedVectors.load('model/word2vec.model')
  train = DL(model_embedding)
  
  # Validate model
  train.loadModel('model/DLmodels.h5', 300)
  print(train.score(X_val, Y_val))

  # Predict
  sen1 = "đồ ăn ngon nhưng chờ rất lâu"
  sen2 = "chờ lâu nhưng đồ ăn rất ngon"
  print(train.predict(sen1))
  print(train.predict(sen2))
  
# pipeline()  
validate()