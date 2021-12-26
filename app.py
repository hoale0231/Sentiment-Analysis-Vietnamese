from tkinter import Tk, BOTH, Text, X, TOP, BOTTOM, LEFT, RIGHT, END, Frame, Button, Label
from tkinter.ttk import Style
from DL import DL
from prep import text_preprocess
import tkinter.messagebox as mbox
import numpy as np
import json
import gensim.models as word2vec


class App(Frame):
    def __init__(self, parent):
      Frame.__init__(self, parent)
  
      self.parent = parent
      self.initUI()

      # For Deep Learning
      self.model_embedding = word2vec.KeyedVectors.load('./model/word2vec.model')
      self.train_DL = DL(self.model_embedding)
      self.train_DL.loadModel('./model/DLmodels.h5', 300)
  
    def initUI(self):
      self.parent.title("Phân tích cảm xúc")
      self.pack(fill=BOTH, expand=True)
  
      Style().configure("TFrame", background="#fff")
  
      frame_1 = Frame(self)
      frame_1.pack(fill=X)
      label_1 = Label(frame_1, text="Nhập bình luận: ", font=("Calibri", 13))
      label_1.pack(side=LEFT, padx=5, pady=5)
      self.txt = Text(frame_1, height=5)
      self.txt.pack(fill=X, padx=5, pady=5, expand=True)
      frame_2 = Frame(self)
      frame_2.pack(fill=X)
      label_2 = Label(frame_2, text="Đây là bình luận: ", font=("Calibri", 13))
      label_2.pack(side=LEFT, padx=5, pady=5)
      self.label_result = Label(frame_2, text="", font=("Calibri", 16))
      self.label_result.pack(side=LEFT, padx=5, pady=5)

      logistic_button = Button(self, text="Logistic Regression", width=18, font=("Calibri", 12), activebackground="orange", command=self.predict_logistic)
      logistic_button.place(x=0, y=150)

      bayes_button = Button(self, text="Naive Bayes", width=18, font=("Calibri", 12), activebackground="orange", command=self.predict_bayes)
      bayes_button.place(x=172, y=150)

      deep_button = Button(self, text="Deep Learning", font=("Calibri", 12), width=18, activebackground="orange", command=self.predict_deep)
      deep_button.place(x=344, y=150)

      frame2 = Frame(self)
      frame2.pack(fill=X, side=BOTTOM)

      reset_button = Button(frame2, text="Reset", width=18, font=("Calibri", 12), command=self.reset)
      reset_button.pack(side=BOTTOM, padx=5, pady=5)
      
      remove_result_button = Button(frame2, text="Remove result", width=18, font=("Calibri", 12), command=self.remove_result)
      remove_result_button.pack(side=BOTTOM, padx=5, pady=5)

    # Function to UI
    def retrieve_input(self):
      input = self.txt.get("1.0", END)
      return input

    def show_positive(self):
      self.label_result.pack(side=LEFT, padx=5, pady=5)
      self.label_result.configure(text="Tích cực", fg="blue")

    def show_negative(self):
      self.label_result.pack(side=LEFT, padx=5, pady=5)
      self.label_result.configure(text="Tiêu cực", fg="red")

    def remove_result(self):
      self.label_result.pack_forget()

    def reset(self):
      self.txt.delete("1.0", END)
      self.label_result.pack_forget()
    
    def notification(self):
      mbox.showwarning("Warning", "Please input comment!")

    # Predict Logistic Regression
    def sigmoid(self, s):
      return 1/(1 + np.exp(-s))

    def prob(self, w, X):
      """
      X: a 2d numpy array of shape (N, d). N datatpoint, each with size d
      w: a 1d numpy array of shape (d)
      """
      return self.sigmoid(X.dot(w))

    def predict_logistic(self):
      text_input = self.retrieve_input()
      if (len(text_input) == 1):
        self.notification()
        return
      text_input = text_preprocess(text_input)
      w_load = np.load('./model/LR_w.npy').tolist()
      listword_load = np.load('./model/LR_listword.npy').tolist()
      w_load = np.array([w_load])
      bow_text = text_input.split(" ")
      wordDictText = dict.fromkeys(listword_load, 0)
      for word in bow_text:
        try:
          wordDictText[word] += 1
        except:
          continue
      X_text = []
      for word in wordDictText:
        X_text.append(wordDictText[word])
      X_text_np = np.array([X_text])
      X_text = None
      X_text_np = np.concatenate((np.ones((X_text_np.shape[0], 1)), X_text_np), axis = 1)
      result = self.prob(w_load.T, X_text_np)
      print("Probability Logistic Regression: ", result[0, 0])
      if result[0, 0] >= 0.5:
        self.show_positive()
      else:
        self.show_negative()

    # Predict Naive Bayes
    def loadVocab(self):
      fileVocabs = open("./model/vocabs.json", "r", encoding="utf-8")
      self.vocabs = json.load(fileVocabs)

    def load(self, priors, likelihoods, classes):
      self._priors = priors
      self._likelihoods = likelihoods
      self._classes = classes

    # Change samples to vector
    # parameter single: True when X is a single sample else False
    def feature_extract_func(self, X, single=False):
      if single:
        X_new = np.zeros((len(self.vocabs), ))
        words = X.split(' ')
        for word in words:
          if word in self.vocabs.keys():
            X_new[self.vocabs[word]] += 1
      else:
        X_new = np.zeros((X.shape[0], len(self.vocabs)))
        for i, sample in enumerate(X):
          words = sample.split(' ')
          for word in words:
            if word in self.vocabs.keys():
              X_new[i][self.vocabs[word]] += 1

      return X_new

    def calc_likelihood(self, cls_likeli, x_test):
      return np.log(cls_likeli) * x_test

    def predict_bayes(self):
      text_input = self.retrieve_input()
      if (len(text_input) == 1):
        self.notification()
        return
      text_input = text_preprocess(text_input)
      priors = np.loadtxt('./model/NB_priors.csv', delimiter=',')
      likelihoods = np.loadtxt('./model/NB_likelihoods.csv', delimiter=',')
      self.loadVocab()
      self.load(priors, likelihoods, np.array([0, 1]))

      # Feature extraction
      x_test = self.feature_extract_func(text_input, single=True)

      # Calculate posterior for each class
      posteriors = []
      for idx, c in enumerate(self._classes):
          prior_c = np.log(self._priors[idx])
          likelihoods_c = self.calc_likelihood(self._likelihoods[idx,:], x_test)
          posteriors_c = np.sum(likelihoods_c) + prior_c
          posteriors.append(posteriors_c)

      pred_idx = np.argmax(posteriors)
      if (self._classes[pred_idx] == 1):
        self.show_positive()
      else:
        self.show_negative()

    # Predict Deep Learning
    def predict_deep(self):
      text_input = self.retrieve_input()
      if (len(text_input) == 1):
        self.notification()
        return
      result = self.train_DL.predict(text_input)
      if (result == "Positive"):
        self.show_positive()
      else:
        self.show_negative()
       
root = Tk()
root.geometry("500x300+500+250")
app = App(root)
root.mainloop()