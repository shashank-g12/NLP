from transformers import AutoTokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader
import torch
import json
import transformers
import re
import os
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cache_dir = 't5-small'
model_path = 't5-acc-2.pth'
tokenizer = AutoTokenizer.from_pretrained(cache_dir)

train_path = sys.argv[1]
val_path = sys.argv[2]

EPOCHS = 35
BATCH_SIZE = 64

def read_data(file_path):
	data = []
	with open(file_path, 'r') as fd:
		json_list = list(fd)
		for json_str in json_list:
			result = json.loads(json_str)
			data.append(result)
	return data

def preprocess_text(input_text):
	# Remove disfluencies from the input text
	input_text = re.sub(r'\b([uU]h+|[uU]m+|[uU]hm+|yeah|oh+|Oh+)\b', '', input_text)
	input_text = re.sub(r'\b(\w+(?:\s+\w+)*)\s+(\1)\b', r'\1', input_text)
	tokens = input_text.split()
	# Remove consecutive repeated words
	tokens = [token for i, token in enumerate(tokens) if i == 0 or token != tokens[i-1]]
	# Convert tokens back to a string
	input_text = ' '.join(tokens)
	input_text = re.sub(r'\b(\w+(?:\s+\w+)*)\s+(\1)\b', r'\1', input_text)
	return input_text

def process_data(data):
	data_x, data_y = [], []
	for sample in data:
		if not sample['history']:
			inp = preprocess_text(sample['input'])+ ' contacts: ' + ' '.join(sample['user_contacts'])
			out = sample['output']
		else:
			query = sample['history'][0]['user_query']
			response = sample['history'][0]['response_text']
			inp = preprocess_text(sample['input'])+ ' query: ' + query + ' response: ' + response  + ' contacts: ' + ' '.join(sample['user_contacts'])
			out = sample['output']
		data_x.append(inp)
		data_y.append(out)
	return list(zip(data_x, data_y))


def collate(batch):
	X, Y = zip(*batch)
	X = tokenizer(list(X), padding = True, return_tensors = 'pt')
	Y = tokenizer(list(Y), padding = True, return_tensors = 'pt')
	Y['input_ids'][Y['input_ids']==0] = -100
	return X, Y

train_set = read_data(train_path)
dev_set = read_data(val_path)
train = process_data(train_set)
dev = process_data(dev_set)
train_loader = DataLoader(train, batch_size = BATCH_SIZE, shuffle = True, collate_fn=collate)
dev_loader = DataLoader(dev, batch_size = BATCH_SIZE, collate_fn=collate)


model = T5ForConditionalGeneration.from_pretrained(cache_dir).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
lr_scheduler = transformers.get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,\
											num_training_steps=(50)*len(train_loader))

def val_accuracy(dev_loader, model, dev_set):
	idx = 0
	acc = 0
	model.eval()
	with torch.no_grad():
		for (X, _) in dev_loader:
			input_ids  = X["input_ids"].to(device)
			results = model.generate(input_ids, num_beams=3, min_length=0, max_length=150)
			results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=False)
			for result in results:
				if result==dev_set[idx]['output']: acc+=1
				idx+=1
	return acc/len(dev_set)

def val_loop(dataloader, model):
	size = len(dataloader)
	cost = 0.0
	model.eval()
	with torch.no_grad():
		for X,Y in dataloader:
			input_ids = X['input_ids'].to(device)
			attention_mask = X['attention_mask'].to(device)
			decoder_input_ids =  Y['input_ids'].to(device)
			output = model(input_ids, attention_mask, labels = decoder_input_ids)
			cost += output.loss.item()
	return cost/size

def train_model(train_loader, model):
	step_check = 50
	step_count = 0
	val_best = float('inf')
	val_best_acc = -float('inf')
	running_loss = 0.0
	for epoch in range(EPOCHS):
		print(f'\nEpoch {epoch+1}:')
		print('-'*20, flush=True)
		for batch, (X,Y) in enumerate(train_loader):
			model.train()
			optimizer.zero_grad()
			
			input_ids = X['input_ids'].to(device)
			attention_mask = X['attention_mask'].to(device)
			decoder_input_ids = Y['input_ids'].to(device)
			outputs = model(input_ids, attention_mask, labels=decoder_input_ids)
			loss = outputs.loss
			running_loss += loss.item()
			loss.backward()
			optimizer.step()
			lr_scheduler.step()
			
			step_count+=1
			if step_count==step_check:
				print(f'Train_loss: {(running_loss/step_count):5f}', flush=True)
				val_loss = val_loop(dev_loader, model)
				if val_loss < val_best:
					print(f'Val loss decreased: {val_best:7f} ----> {val_loss:7f}', flush=True)
					val_best = val_loss
				step_count = 0
				running_loss = 0.0
		if epoch<5: continue
		val_acc = val_accuracy(dev_loader, model, dev_set)
		print(f'Validation accuracy: {val_acc:7f}', flush=True)
		if val_acc > val_best_acc:
			val_best_acc = val_acc
			torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
	train_model(train_loader, model)