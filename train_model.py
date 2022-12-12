import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import random 

import evaluate

from transformers import EncoderDecoderModel, BertTokenizer, GPT2Tokenizer, get_scheduler,  BertLMHeadModel

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
        self.relu1 = torch.nn.LeakyReLU()
        self.lin2 = torch.nn.Linear(middle_dim, 50)
        self.relu2 = torch.nn.LeakyReLU()
        self.lin3 = torch.nn.Linear(50, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.lin1.weight.data.uniform_(-initrange, initrange)
        self.lin1.bias.data.zero_()
        self.lin2.weight.data.uniform_(-initrange, initrange)
        self.lin2.bias.data.zero_()
        self.lin3.weight.data.uniform_(-initrange, initrange)
        self.lin3.bias.data.zero_()

    def forward(self, hidden_outputs):
        return self.lin3(self.relu2(self.lin2(self.relu1(self.lin1(torch.flatten(hidden_outputs,1,2))))))

# fast gradient iterative method from paper Wang et al 2019
def fgim_attack(model, classifier, target_class, origen_data):
    i = 0
    data = Variable(origen_data.data.clone(), requires_grad=True)
    epsilon = 0.1 # modify and play with this
    cls_criterion = torch.nn.CrossEntropyLoss() # torch.nn.BCELoss(size_average=True) # 
    sentence_og = model.cls(data)
    sentence_og.detach()
    while True:
        # to_var? what do it do?
        # print(data.requires_grad)
        # print(data.shape)
        # print(data)
        output = classifier(data)
        sentence_now = model.cls(data)
        tmp = ((sentence_og - sentence_now)/(sentence_og - sentence_now))
        L_BOW = torch.sum(tmp[not (tmp == 1)])
        L_CLS = cls_criterion(output, target_class)
        loss = L_BOW + L_CLS
        classifier.zero_grad()
        data.retain_grad()
        loss.backward(retain_graph=True)
        # print(loss)
        if data.grad is None:
            break
        data_grad = data.grad.data
        data = data - epsilon*data_grad
        i += 1
        epsilon = epsilon*0.9
        # print(data.shape)
        # print(data)
        if i >= 5:
            break

    return data

# custom dataset class to load data from the .txt files
class BrownStyleDataset(Dataset):
    def __init__(self, stage='train', input_tokenizer=None, output_tokenizer=None):

        # store key variables
        # self.id_to_label = ['adventure', 
        #                     'belles_lettres', 
        #                     'editorial', 
        #                     'fiction', 
        #                     'government', 
        #                     'hobbies', 
        #                     'humor', 
        #                     'learned', 
        #                     'lore', 
        #                     'mystery', 
        #                     'news', 
        #                     'religion', 
        #                     'reviews', 
        #                     'romance', 
        #                     'science_fiction'
        #                     ]
        self.id_to_label = ['adventure', 'editorial']
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

            ot = input_tokenizer(sentences, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
            self.sentences_out = torch.roll(ot.input_ids.to(device), shifts=(-1), dims=(0))
            self.attention_masks_out = ot.attention_mask.to(device)

            # from https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16

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

def load_data(input_tokenizer, output_tokenizer, params):

    # make a bunch of datasets and dataloaders therefrom
    train_dataset = BrownStyleDataset(stage='train', input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    print(f"loaded {len(train_dataloader)} training samples")
    
    val_dataset = BrownStyleDataset(stage='validation', input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True)
    print(f"loaded {len(val_dataloader)} validation samples")
   
    test_dataset = BrownStyleDataset(stage='test', input_tokenizer=input_tokenizer, output_tokenizer=output_tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
    print(f"loaded {len(test_dataloader)} test samples")
    
    return train_dataloader, val_dataloader, test_dataloader

def evaluate_transfer(model, classifier, train_dataloader, eval_dataloader, params, input_tokenizer, output_tokenizer):
    print("Begin testing style transfer!")
    
    metric = evaluate.load("accuracy")
    model.eval()
    classifier.eval()
    pred = []
    truth = []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(input_ids=batch["input_sentences"], labels=batch["output_sentences"], output_hidden_states=True)

        z = outputs.hidden_states[12]

        cls_outputs = classifier(z)
        cls_pred = cls_outputs.argmax(1)
        cls_truth = batch['genre_labels']

        z_alt = fgim_attack(model, classifier, ((cls_pred + 1)%2), z)
        
        # outputs_alt = model(input_ids=batch["input_sentences"], encoder_hidden_states=z_alt)
        logits_alt = model.cls(z_alt)
        
        guess = torch.argmax(outputs.logits, dim=2).long()
        pred = output_tokenizer.batch_decode(guess, skip_special_tokens=True)
        guess_alt = torch.argmax(logits_alt, dim=2).long()
        pred_alt = output_tokenizer.batch_decode(guess_alt, skip_special_tokens=True)
        truth = input_tokenizer.batch_decode(batch['input_sentences'], skip_special_tokens=True)

        print("---------------------------")
        print("example input sentences: ")
        print(truth[0:5])
        print("---------------------------")
        print("example initial output sentences: ")
        print(pred[0:5])
        print("example altered output sentences: ")
        print(pred_alt[0:5])
        print("---------------------------")
        print("true class: ")
        print(cls_truth[0:5])
        print("---------------------------")
        print("predicted class: ")
        print(cls_pred[0:5])
        print("---------------------------")
        print("===========================",flush=True)

def train_classifier(model, classifier, train_dataloader, eval_dataloader, params, input_tokenizer, output_tokenizer):
    print("Begin training classifier!")
    
    num_training_steps = params.num_epochs * len(train_dataloader)

    cls_optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)
    cls_lr_scheduler = get_scheduler(
        name="linear", 
        optimizer=cls_optimizer, 
        num_warmup_steps=0.1*num_training_steps, 
        num_training_steps=num_training_steps
    )
    cls_criterion = torch.nn.CrossEntropyLoss() # torch.nn.BCELoss(size_average=True) # 
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(params.num_epochs):

        model.eval()
        classifier.train()
        for batch in train_dataloader:
            outputs = model(input_ids=batch["input_sentences"], labels=batch["output_sentences"], output_hidden_states=True)
            z = outputs.hidden_states[12]
            cls_outputs = classifier(z)
            # print(z.shape)
            # print(batch["genre_labels"].shape)
            cls_loss = cls_criterion(cls_outputs, batch["genre_labels"])
            cls_loss.backward()
            cls_optimizer.step()
            cls_lr_scheduler.step()
            cls_optimizer.zero_grad()
            progress_bar.update(1)
        print("===========================")
        print(f'epoch {epoch + 1}/{params.num_epochs} | loss: {cls_loss.item()}')
        
        metric = evaluate.load("accuracy")
        model.eval()
        classifier.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(input_ids=batch["input_sentences"], labels=batch["output_sentences"], output_hidden_states=True)
            
            z = outputs.hidden_states[12]
            cls_outputs = classifier(z)
            cls_pred = cls_outputs.argmax(1)
            cls_truth = batch['genre_labels']

            metric.add_batch(predictions=cls_pred, references=cls_truth)

        print("---------------------------")
        print("example input sentences: ")
        print(cls_truth[0:5])
        print("---------------------------")
        print("example output sentences: ")
        print(cls_pred[0:5])
        print("---------------------------")
        score = metric.compute()
        print('Validation Classifier Accuracy:', score['accuracy'])
        print("===========================",flush=True)

    return classifier

def train(model,  train_dataloader, eval_dataloader, params, input_tokenizer, output_tokenizer):
    print("Begin training autoencoder!")
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
            outputs = model(input_ids=batch["input_sentences"], attention_mask=batch['input_attention_masks'], labels=batch["input_sentences"], output_hidden_states=True)
            z = outputs.hidden_states[12]
           
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
        print('loss:', loss.item())
        print("===========================")
        print(f'epoch {epoch + 1}/{params.num_epochs} | loss: {loss.item()}')
        
        metric = evaluate.load("perplexity", module_type='metric')
        model.eval()
        pred = []
        truth = []
        for batch in eval_dataloader:
            with torch.no_grad():
                outputs = model(input_ids=batch["input_sentences"], attention_mask=batch['input_attention_masks'], labels=batch["input_sentences"], output_hidden_states=True)
            
            guess = torch.argmax(outputs.logits,dim=2).long()
            pred = output_tokenizer.batch_decode(guess)
            truth = input_tokenizer.batch_decode(batch['input_sentences'])

            metric.add_batch(predictions=pred, references=truth)
            

        print("---------------------------")
        print("example input sentences: ")
        print(truth[0:5])
        print("---------------------------")
        print("example output sentences: ")
        print(pred[0:5])
        print("---------------------------")
        score = metric.compute(model_id='gpt2')
        print('Mean Perplexity:', score['mean_perplexity'])
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
    
    input_tokenizer = BertTokenizer.from_pretrained("bert-base-cased", is_decoder=True)

    # # pro tip: https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16
    # def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    #     outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    #     return outputs
    # GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

    output_tokenizer =  input_tokenizer
    input_tokenizer.bos_token = input_tokenizer.cls_token
    input_tokenizer.eos_token = input_tokenizer.sep_token

    # output_tokenizer.pad_token = output_tokenizer.unk_token
    # output_tokenizer.cls_token = input_tokenizer.cls_token

    train_dataloader, eval_dataloader, test_dataloader = load_data(input_tokenizer, input_tokenizer, params)

    if params.train_autoencoder:
        model = BertLMHeadModel.from_pretrained("bert-base-uncased")
        print(model)
        
        # model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")
        print("created model")
        
        model.to(device)
        
        model = train(model, train_dataloader, eval_dataloader, params, input_tokenizer, output_tokenizer)
        # torch.save(model.state_dict(),'models/brown_autoencoder.torch')
        model.save_pretrained('models/brown_autoencoder')
    else:
        model = BertLMHeadModel.from_pretrained(f'./models/{params.model_name}')
        # model.load_state_dict(f'./models/{params.model_name}')
        model.to(device)

    if params.train_classifier:
        classifier = GenreClassifier(768, 256, 2)
        model.to(device)
        classifier.to(device)
        print(classifier)
        classifier = train_classifier(model,classifier, train_dataloader, eval_dataloader, params, input_tokenizer, output_tokenizer)
        torch.save(classifier.state_dict(), 'models/brown_latent_classifier.torch')
    else:
        classifier = GenreClassifier(768, 256, 2)
        classifier.load_state_dict(torch.load(f'models/{params.classifier_name}'))
        classifier.to(device)

    evaluate_transfer(model,classifier, train_dataloader, eval_dataloader, params, input_tokenizer, output_tokenizer)

    if params.test:
        # first load model
        test(model, test_dataloader, input_tokenizer, output_tokenizer)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--train_autoencoder", type=bool, default=False)
    parser.add_argument("--train_classifier", type=bool, default=False)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="brown_autoencoder")
    parser.add_argument("--classifier_name", type=str, default="brown_latent_classifier.torch")

    params, unknown = parser.parse_known_args()
    main(params)








