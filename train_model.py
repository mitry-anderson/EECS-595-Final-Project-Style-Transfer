import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import random 

import evaluate

from transformers import EncoderDecoderModel, BertTokenizer, GPT2Tokenizer, get_scheduler

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

MAX_LENGTH = 100

class GenreClassifier(torch.nn.Module):

    def __init__(self, hidden_dim, middle_dim, num_class):
        super(GenreClassifier, self).__init__()
        self.lin1 = torch.nn.Linear(hidden_dim*MAX_LENGTH, middle_dim)
        self.relu = torch.nn.ReLU()
        self.lin2 = torch.nn.Linear(middle_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.lin1.weight.data.uniform_(-initrange, initrange)
        self.lin1.bias.data.zero_()
        self.lin2.weight.data.uniform_(-initrange, initrange)
        self.lin2.bias.data.zero_()

    def forward(self, hidden_outputs):
        return self.lin2(self.relu(self.lin1(torch.flatten(hidden_outputs,1,2))))

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
            it = input_tokenizer(sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
            self.sentences = it.input_ids.to(device)
            self.attention_masks = it.attention_mask.to(device)

            ot = output_tokenizer(sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
            self.sentences_out = ot.input_ids.to(device)
            self.attention_masks_out = ot.attention_mask.to(device)

            # from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
            self.attention_masks_out = torch.tensor([
                [-100 if mask == 0 else token for mask, token in mask_and_tokens] for mask_and_tokens in [zip(masks, labels) for masks, labels in zip(self.attention_masks_out, self.sentences_out)]
            ], device=device)

            self.labels = torch.tensor(labels,device=device)
        else:
            raise RuntimeError("need to feed the dataset a tokenizer bro")
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            "input_sentences" : self.sentences[idx], 
            "input_attention_masks" : self.attention_masks[idx], 
            "output_sentences" : self.sentences_out[idx], 
            "output_attention_masks" : self.attention_masks_out[idx], 
            "genre_labels" : self.labels[idx]
        }

    # def to(self, device):
    #     self.sentences.to(device)
    #     self.labels.to(device)

def load_data(input_tokenizer, output_tokenizer, params):

    # make a bunch of datasets and dataloaders therefrom
    train_dataset = BrownStyleDataset(stage='train', input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    print(f"loaded {len(train_dataloader)} training samples")
    
    val_dataset = BrownStyleDataset(stage='validation', input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)
    print(f"loaded {len(val_dataloader)} validation samples")
   
    test_dataset = BrownStyleDataset(stage='test', input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
    print(f"loaded {len(test_dataloader)} test samples")
    
    return train_dataloader, val_dataloader, test_dataloader

def train(model, classifier, train_dataloader, eval_dataloader, params, input_tokenizer, output_tokenizer):
    print("Begin training!")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_training_steps = params.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0.1*num_training_steps, 
        num_training_steps=num_training_steps
    )

    cls_optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-5)
    cls_lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=optimizer, 
        num_warmup_steps=0.1*num_training_steps, 
        num_training_steps=num_training_steps
    )
    cls_criterion = torch.nn.CrossEntropyLoss()
    # progress_bar = tqdm(range(num_training_steps))
    for epoch in range(params.num_epochs):

        model.train()
        classifier.train()
        for batch in train_dataloader:
            outputs = model(input_ids=batch["input_sentences"], labels=batch["output_sentences"])
            cls_outputs = classifier(outputs.encoder_last_hidden_state)
            print(f"hidden state: {outputs.encoder_last_hidden_state.shape}")

            print(f"cls_outputs: {cls_outputs.shape}")
            print(f"label: {batch['genre_labels'].shape}")
            print(cls_outputs)
            print(batch['genre_labels'])
            
            loss = outputs.loss
            loss.backward(retain_graph=True)

            z = outputs.encoder_last_hidden_state
            cls_outputs = classifier(z)

            cls_loss = cls_criterion(cls_outputs, batch["genre_labels"])
            cls_loss.backward(retain_graph=True)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            cls_optimizer.step()
            cls_lr_scheduler.step()
            cls_optimizer.zero_grad()
            # progress_bar.update(1)
        # print('loss:', loss.item())
        print("===========================")
        print(f'epoch {epoch + 1}/{params.num_epochs} | loss: {loss.item()}')
        
        metric = evaluate.load("exact_match")
        metric2 = evaluate.load("accuracy")
        model.eval()
        classifier.eval()
        pred = []
        truth = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model.generate(input_ids=batch['input_sentences'], return_dict_in_generate=True, output_hidden_states=True)
            
            z = outputs.encoder_hidden_states[0]
            cls_outputs = classifier(z)
            
            pred = output_tokenizer.batch_decode(outputs.sequences)
            truth = input_tokenizer.batch_decode(batch['input_sentences'])

            metric.add_batch(predictions=pred, references=truth)
            metric2.add_batch(predictions=cls_outputs.argmax(1), references=batch['genre_labels'])

        print("---------------------------")
        print("example input sentences: ")
        print(truth[0:3])
        print("---------------------------")
        print("example output sentences: ")
        print(pred[0:3])
        print("---------------------------")
        score = metric.compute()
        print('Validation Accuracy:', score['exact_match'])
        print("===========================",flush=True)
    return model

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
    
    input_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # pro tip: https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs
    GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

    output_tokenizer =  GPT2Tokenizer.from_pretrained('gpt2')
    input_tokenizer.bos_token = input_tokenizer.cls_token
    input_tokenizer.eos_token = input_tokenizer.sep_token

    output_tokenizer.pad_token = output_tokenizer.unk_token
    # output_tokenizer.cls_token = input_tokenizer.cls_token

    train_dataloader, eval_dataloader, test_dataloader = load_data(input_tokenizer, output_tokenizer, params)

    if params.train:
        model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "gpt2")
        print(model)
        classifier = GenreClassifier(768, 256, 2)
        # model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
        print("created model")
        model.decoder.config.use_cache = False
        model.config.decoder_start_token_id = output_tokenizer.bos_token_id
        model.config.eos_token_id = output_tokenizer.eos_token_id
        # model.config.max_length = MAX_LENGTH
        model.config.max_new_tokens = MAX_LENGTH
        model.config.no_repeat_ngram_size = 3
        model.early_stopping = True
        model.config.pad_token_id = input_tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        
        model.to(device)
        classifier.to(device)
        
        model, classifier = train(model, classifier, train_dataloader, eval_dataloader, params, input_tokenizer, output_tokenizer)
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








