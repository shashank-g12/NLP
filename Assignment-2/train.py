import numpy as np
import torch
import sklearn.metrics as metrics
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import os,sys
import pickle

train_file_path = sys.argv[1]
val_file_path = sys.argv[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_file(file_path):
	with open(file_path) as fd:
		sentences = []
		labels = []
		text = ''
		label = ''
		for line in fd.readlines():
			if line == '\n':
				sentences.append(text)
				labels.append(label)
				text = ''
				label = ''
				continue
			text += line.split('\t')[0] + ' '
			label += line.split('\t')[1].split('\n')[0] + ' '
		if text!='':
			sentences.append(text)
			labels.append(label)
	return sentences, labels

def build_vocab(sentences, specials=None):
	count_dict = {}
	for sentence in sentences:
		for word in sentence.split():
			if word in count_dict:
				count_dict[word] += 1
			else:
				count_dict[word] = 1
	item_list = list(count_dict.items())
	item_list.sort(reverse = True, key = (lambda items: items[1]))
	count_dict = dict(item_list)
	return vocab(count_dict, specials = specials)

def data_process(sentences, labels, data_vocab, class_vocab):
	data = []
	for sentence, label in zip(sentences, labels):
		tensor_x = torch.tensor([data_vocab[token] for token in sentence.split()], dtype = torch.long)
		tensor_y = torch.tensor([class_vocab[clas] for clas in label.split()], dtype = torch.long)
		data.append((tensor_x, tensor_y))
	return data

def generate_batch(data_batch):
	X, Y = [], []
	for sentence, label in data_batch:
		X.append(sentence)
		Y.append(label)
	X = pad_sequence(X, batch_first = True, padding_value=data_vocab['<pad>'])
	Y = pad_sequence(Y, batch_first = True, padding_value=class_vocab['O'])
	return X,Y

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


def train_loop(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X = X.to(device)
		y = y.to(device)
		logits = model(X).permute(0,2,1)
		loss = loss_fn(logits, y)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		if batch % 100 == 0:
			loss_, current = loss.item(), min((batch + 1) * BATCH_SIZE, size)
			print(f"loss: {loss_:7f}  [{current}/{size}]")

def val_loop(dataset, model, loss_fn):
	true_tag = []
	pred_tag = []
	cost = 0.0
	size = len(dataset)
	with torch.no_grad():
		model.eval()
		for data in dataset:
			x = data[0].to(device)
			y = data[1].to(device)
			prob = model(x)
			loss = loss_fn(prob, y)
			cost += loss.item()
			pred = torch.argmax(prob, dim=1)
			
			pred_tag.extend(pred.tolist())
			true_tag.extend(y.tolist())
		
		print(f"val loss: {cost/size}")
		possible_labels = list(class_vocab.get_stoi().values())
		possible_labels.remove(class_vocab['O'])
		f1_micro = metrics.f1_score(true_tag, pred_tag, average="micro", labels=possible_labels)
		f1_macro = metrics.f1_score(true_tag, pred_tag, average="macro", labels=possible_labels)
		print(f"Average micro and macro: {round((f1_micro+f1_macro)/2, 5)}")
	return cost/size

train_sentences, train_labels = read_file(train_file_path)
val_sentences, val_labels = read_file(val_file_path)
data_vocab =  build_vocab(train_sentences, specials=['<unk>', '<pad>'])
data_vocab.set_default_index(data_vocab['<unk>'])
class_vocab = build_vocab(train_labels)

with open('vocab_file.pkl', 'wb') as fd:
	pickle.dump((data_vocab, class_vocab), fd)

train_data = data_process(train_sentences, train_labels, data_vocab, class_vocab)
val_data = data_process(val_sentences, val_labels, data_vocab, class_vocab)

BATCH_SIZE = 128
PATIENCE = 5
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
						shuffle=True, collate_fn=generate_batch)

model = BiLSTM(input_dim=len(data_vocab), hid_dim=512, emb_dim = 300).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_loss=10000
patience_count=0
for epoch in range(25):
	print(f"Epoch: {epoch+1}")
	train_loop(train_loader, model, loss_fn, optimizer)
	curr_loss = val_loop(val_data, model, loss_fn)
	if curr_loss<best_loss:
		best_loss=curr_loss
		torch.save(model.state_dict(), 'aib222684_model')
		patience_count=0
	else:
		patience_count+=1
	if patience_count==PATIENCE:
		break