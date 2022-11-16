from transformers import EncoderDecoderModel, BertTokenizer, GPT2TokenizerFast
import torch

input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
output_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "gpt2")

model.config.decoder_start_token_id = input_tokenizer.cls_token_id
model.config.pad_token_id = input_tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

input_ids = input_tokenizer("This is a really long text", return_tensors="pt").input_ids
labels = input_tokenizer("This is a really long text", return_tensors="pt").input_ids
outputs = model(input_ids=input_ids, labels=input_ids)
loss, logits = outputs.loss, outputs.logits

print(input_ids)
print(loss)
print(outputs.logits.shape)
guess = torch.max(torch.softmax(outputs.logits,dim=2),dim=2)[0].squeeze().long()
print(guess)
print(output_tokenizer.decode(guess))
