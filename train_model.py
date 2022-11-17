import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import random 

import evaluate

from transformers import EncoderDecoderModel, BertTokenizer, GPT2TokenizerFast, get_scheduler

from tqdm.auto import tqdm

import logging

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")

# custom dataset class to load data from the .txt files
class BrownStyleDataset(Dataset):
    def __init__(self, stage='train', input_tokenizer=None, output_tokenizer=None):

        # store key variables
        self.id_to_label = ['adventure', 'news']
        self.stage = stage
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer

        # load data to tensors
        sentences = []
        labels = []
        for i in range(len(self.id_to_label)):
            with open(f'dataset/{self.id_to_label[i]}_{self.stage}.txt') as file:
                for sentence in file:
                    sentences.append(sentence)
                    labels.append(i)
        
        # if applicable (should always be!), tokenize data
        if self.input_tokenizer is not None and self.output_tokenizer is not None:
            self.sentences = input_tokenizer(sentences, return_tensors="pt", padding=True).input_ids.to(device)
            self.sentences_out = output_tokenizer(sentences, return_tensors="pt", padding=True).input_ids.to(device)

            self.labels = torch.tensor(labels,device=device)
        else:
            raise RuntimeError("need to feed the dataset a tokenizer bro")
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.sentences_out[idx], self.labels[idx]

    # def to(self, device):
    #     self.sentences.to(device)
    #     self.labels.to(device)

def load_data(input_tokenizer, output_tokenizer, params):

    # make a bunch of datasets and dataloaders therefrom
    train_dataset = BrownStyleDataset(stage='train', input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    print(f"loaded {len(train_dataloader)} training samples")
    
    val_dataset = BrownStyleDataset(stage='validation', input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True)
    print(f"loaded {len(val_dataloader)} validation samples")
   
    test_dataset = BrownStyleDataset(stage='test', input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)
    print(f"loaded {len(test_dataloader)} test samples")
    
    return train_dataloader, val_dataloader, test_dataloader

def train(model, train_dataloader, eval_dataloader, params, input_tokenizer, output_tokenizer):
    print("Begin training!")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_training_steps = params.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0.1*num_training_steps, 
        num_training_steps=num_training_steps
    )
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(params.num_epochs):

        model.train()
        for batch in train_dataloader:
            outputs = model(input_ids=batch[0], labels=batch[1])
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        print('loss:', loss.item())
        
        metric = evaluate.load("exact_match")
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model.generate(input_ids=batch[0], max_new_tokens=50)

            pred = []
            truth = []
            print(outputs)
            for i in range(len(outputs)):
                pred.append(output_tokenizer.decode(outputs[i], skip_special_tokens=True))
                truth.append(input_tokenizer.decode(batch[0][i], skip_special_tokens=True))
            metric.add_batch(predictions=pred, references=truth)

            print("===========================")
            print("input sentence: ")
            print(truth[0])
            print("output sentence: ")
            print(pred[0])
            print("===========================")
        
        score = metric.compute()
        print('Validation Accuracy:', score['exact_match'])

def test(model, test_dataloader, input_tokenizer, output_tokenizer):
    metric = evaluate.load("exact_match")
    model.eval()
    for batch in test_dataloader:
        with torch.no_grad():
                outputs = model.generate(input_ids=batch[0])
        pred = []
        truth = []
        for i in range(len(outputs)):
            pred.append(output_tokenizer.decode(outputs[i]))
            truth.append(input_tokenizer.decode(batch[0][i]))
        metric.add_batch(predictions=pred, references=truth)
    
    score = metric.compute()
    print('Validation Accuracy:', score['exact_match'])

def main(params):
    
    input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    output_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    output_tokenizer.pad_token = input_tokenizer.pad_token
    output_tokenizer.cls_token = input_tokenizer.cls_token

    train_dataloader, eval_dataloader, test_dataloader = load_data(input_tokenizer, output_tokenizer, params)

    if params.train:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "gpt2")
        print("created model")
        model.config.decoder_start_token_id = input_tokenizer.cls_token_id
        model.config.pad_token_id = input_tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.to(device)
        model = train(model, train_dataloader, eval_dataloader, params, input_tokenizer, output_tokenizer)
        model.save_pretrained('models/news_adventure.torch')
    else:
        model = EncoderDecoderModel.from_pretrained(f'models/{params.model_name}')

    if params.test:
        # first load model
        test(model, test_dataloader, input_tokenizer, output_tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="news_adventure.torch")

    params, unknown = parser.parse_known_args()
    main(params)








