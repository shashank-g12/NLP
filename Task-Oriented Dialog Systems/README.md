# Dialog Parsing for Task-Oriented Dialog Systems

- A task-oriented dialog (TOD) system assists the user in achieving goals like managing the calendar, sending emails, controlling devices, etc., through natural language. A crucial task inthe TOD system is to extract meaning (intent) from the user input and its associated information (slot-values) which is the focus of this assignment. Specifically, we will develop a parsing model which takes input a user utterance, dialog history, and extra context information and predicts the parsed output.

- Finetuned T5-small on the dataset to give the parsed output which gives an exact match accuracy of 87%.

## Setup

- Install pytorch (v1.13) and Huggingface transformers (v4.21)

## Running the code

- finetune the model by using the command
```
bash run_model.sh train <train-file> <val-file>
```

- `<train-file>`: A jsonl file containing the user input and the parsed output
- `<val-file>`: A jsonl file containing the user input and the parsed output

- Predict the output for the test set using the command
```
bash run_model.sh test <test-file> outfile.txt
```

This generates a file `outfile.txt` which contains the predicted output.
