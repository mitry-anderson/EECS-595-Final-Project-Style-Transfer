import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import random 

import evaluate

from transformers import EncoderDecoderModel, BertTokenizer, GPT2TokenizerFast, get_scheduler

from tqdm.auto import tqdm


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
    def __init__(self, stage='train', tokenizer=None):

        # store key variables
        self.id_to_label = ['adventure', 'news']
        self.stage = stage
        self.tokenizer = tokenizer

        # load data to tensors
        sentences = []
        labels = []
        for i in range(len(self.id_to_label)):
            with open(f'dataset/{self.id_to_label[i]}_{self.stage}.txt') as file:
                for sentence in file:
                    sentences.append(sentence)
                    labels.append(i)
        
        # if applicable (should always be!), tokenize data
        if self.tokenizer is not None:
            input_ids = tokenizer(sentences, return_tensors="pt", padding=True).input_ids
            self.sentences = torch.tensor(input_ids, device=device)
            self.labels = torch.tensor(labels, device=device)
        else:
            self.sentences = sentences
            self.labels = labels
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    # def to(self, device):
    #     self.sentences.to(device)
    #     self.labels.to(device)

def load_data(tokenizer, params):

    # make a bunch of datasets and dataloaders therefrom
    train_dataset = BrownStyleDataset(stage='train',tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    print(f"loaded {len(train_dataloader)} training samples")
    
    val_dataset = BrownStyleDataset(stage='validation',tokenizer=tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True)
    print(f"loaded {len(val_dataloader)} validation samples")
   
    test_dataset = BrownStyleDataset(stage='test',tokenizer=tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=True)
    print(f"loaded {len(test_dataloader)} test samples")
    
    return train_dataloader, val_dataloader, test_dataloader

def train(model, train_dataloader, eval_dataloader, params):
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

            outputs = model(input_ids=batch[0], labels=batch[0])
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            print(loss.item())

        metric = evaluate.load("accuracy")
        model.eval()

        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        
        score = metric.compute()
        print('Validation Accuracy:', score['accuracy'])

def test(model, test_dataloader):
    pass

def main(params):
    
    input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    output_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    train_dataloader, eval_dataloader, test_dataloader = load_data(input_tokenizer, params)

    if params.train:

        model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "gpt2")
        print("created model")
        model.config.decoder_start_token_id = input_tokenizer.cls_token_id
        model.config.pad_token_id = input_tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        model.to(device)
        model = train(model, train_dataloader, eval_dataloader, params)

    if params.test:
        # first load model
        test(model, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)

    params, unknown = parser.parse_known_args()
    main(params)








