# from transformers import EncoderDecoderModel, BertTokenizer, GPT2TokenizerFast
# import torch
# import evaluate

# input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# output_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
# # output_tokenizer.pad_token = input_tokenizer.pad_token
# # output_tokenizer.cls_token = input_tokenizer.cls_token

# model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

# model.config.decoder_start_token_id = input_tokenizer.cls_token_id
# model.config.pad_token_id = input_tokenizer.pad_token_id
# model.config.vocab_size = model.config.decoder.vocab_size

# input_ids = input_tokenizer(["This is a really long text","so is this","so is this"], return_tensors="pt", padding=True).input_ids
# labels = input_tokenizer(["This is a really long text","so is this","so is this"], return_tensors="pt", padding=True).input_ids
# # outputs = model(input_ids=input_ids, labels=input_ids)
# # loss, logits = outputs.loss, outputs.logits

# # print(input_ids)
# # print(loss)
# # print(outputs.logits.shape)
# # guess = torch.argmax(outputs.logits,dim=2).long()
# # print(guess)
# # print(output_tokenizer.decode(guess))



# metric = evaluate.load("exact_match")
# model.eval()
# with torch.no_grad():
#     outputs = model.generate(input_ids=input_ids)

# pred = []
# truth = []
# for i in range(len(outputs)):
#     pred.append(output_tokenizer.decode(outputs[i]))
#     truth.append(input_tokenizer.decode(input_ids[i]))

# print("======================")
# print(pred)
# print("======================")
# print(truth)

# metric.add_batch(predictions=pred, references=truth)
# score = metric.compute()
# print('Validation Accuracy:', score['exact_match'])







# from transformers import EncoderDecoderModel, BertTokenizer
# import torch

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = EncoderDecoderModel.from_encoder_decoder_pretrained(
#     "bert-base-uncased", "bert-base-uncased"
# )  # initialize Bert2Bert from pre-trained checkpoints

# # training
# model.config.decoder_start_token_id = tokenizer.cls_token_id
# model.config.pad_token_id = tokenizer.pad_token_id
# model.config.vocab_size = model.config.decoder.vocab_size

# input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
# labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
# outputs = model(input_ids=input_ids, labels=input_ids)
# loss, logits = outputs.loss, outputs.logits

# # save and load from pretrained
# model.save_pretrained("bert2bert")
# model = EncoderDecoderModel.from_pretrained("bert2bert")

# # generation
# generated = model.generate(input_ids)
# print(input_ids)
# print(labels)
# print(generated)
# print(tokenizer.batch_decode(generated))



import torch
from transformers import BertTokenizer, BertLMHeadModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertLMHeadModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Cappy looked wary , but he moved off the floorboards and followed the dirty ex-musician to the center of the refuse-littered boxcar .", return_tensors="pt")
outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)
print(outputs.loss)
guess = torch.argmax(outputs.logits,dim=2).long()
# print(guess)
print(tokenizer.batch_decode(guess))

z_new = outputs.hidden_states[12] + (torch.zeros_like(outputs.hidden_states[12]) + 1)
prediction_scores = model.cls(z_new)
guess2 = torch.argmax(prediction_scores,dim=2).long()
# print(guess2)
print(tokenizer.batch_decode(guess2))















