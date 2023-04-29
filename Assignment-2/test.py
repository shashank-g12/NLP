import numpy as np
import torch
import sklearn.metrics as metrics
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import os, sys
import pickle

test_file_path = sys.argv[1]
out_file_path = sys.argv[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_file(file_path):
	with open(file_path) as fd:
		sentences = []
		text = ''
		for line in fd.readlines():
			if line == '\n':
				sentences.append(text)
				text = ''
				continue
			text += line.split('\n')[0] + ' '
		if text!='':
			sentences.append(text)
	return sentences


def data_process(sentences, data_vocab):
	data = []
	for sentence in sentences:
		tensor_x = torch.tensor([data_vocab[token] for token in sentence.split()], dtype = torch.long)
		data.append(tensor_x)
	return data

def get_predictions(dataset, model, filename):
	with torch.no_grad():
		fd = open(filename, 'w')
		model.eval()
		for data in dataset:
			x = data.to(device)
			prob = model(x)
			pred = torch.argmax(prob, dim=1)
			pred = pred.tolist()
			for j in range(len(pred)):
				fd.write(class_dict[pred[j]]+'\n')
			fd.write('\n')
			
class BiLSTM(torch.nn.Module):
	def __init__(self, input_dim, hid_dim, emb_dim):
		super(BiLSTM, self).__init__()
		self.input_dim = input_dim
		self.hid_dim = hid_dim
		self.emb_dim = emb_dim
		self.tagset_size = len(class_vocab)
		
		self.embedding = torch.nn.Embedding(input_dim, emb_dim)
		self.lstm = torch.nn.LSTM(emb_dim, hid_dim, num_layers = 1, bidirectional=True, batch_first=True)
		self.fc = torch.nn.Linear(2*hid_dim, self.tagset_size)
	
	def forward(self, x):
		embedded = self.embedding(x)
		outputs, _ = self.lstm(embedded)
		logits = self.fc(outputs)
		return logits

with open('vocab_file.pkl', 'rb') as fd:
	(data_vocab,class_vocab) = pickle.load(fd)

test_sentences = read_file(test_file_path)
test_data = data_process(test_sentences, data_vocab)
model_test = BiLSTM(input_dim=len(data_vocab), hid_dim=512, emb_dim = 300).to(device)
model_test.load_state_dict(torch.load('aib222684_model'))
class_dict = class_vocab.get_itos()

get_predictions(test_data, model_test, out_file_path)