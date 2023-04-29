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
test_path = sys.argv[1]
outfile = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(cache_dir)
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
	data_x = []
	for sample in data:
		if not sample['history']:
			inp = preprocess_text(sample['input'])+ ' contacts: ' + ' '.join(sample['user_contacts'])
		else:
			query = sample['history'][0]['user_query']
			response = sample['history'][0]['response_text']
			inp = preprocess_text(sample['input'])+ ' query: ' + query + ' response: ' + response  + ' contacts: ' + ' '.join(sample['user_contacts'])
		data_x.append(inp)
	return list(data_x)


def collate(batch):
	X = batch
	X = tokenizer(list(X), padding = True, return_tensors = 'pt')
	return X

test_set = read_data(test_path)
test = process_data(test_set)
test_loader = DataLoader(test, batch_size = BATCH_SIZE, collate_fn=collate)

model_test = T5ForConditionalGeneration.from_pretrained(cache_dir).to(device)
model_test.load_state_dict(torch.load(model_path))

def generate_file(fileName, dataloader, model):
	fd = open(fileName, 'w+')
	model.eval()
	with torch.no_grad():
		for X in dataloader:
			input_ids  = X["input_ids"].to(device)
			results = model.generate(input_ids, num_beams=5, min_length=0, max_length=150)
			results = tokenizer.batch_decode(results, skip_special_tokens=True, clean_up_tokenization_spaces=False)
			for result in results:
				fd.write(result+'\n')
	fd.close()

if __name__ == "__main__":
	print(f"generating file....", flush=True)
	generate_file(outfile, test_loader, model_test)
	print(f"done.")