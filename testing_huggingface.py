from transformers import EncoderDecoderModel, BertTokenizer, GPT2TokenizerFast
import torch
import evaluate

input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
output_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
# output_tokenizer.pad_token = input_tokenizer.pad_token
# output_tokenizer.cls_token = input_tokenizer.cls_token

model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

model.config.decoder_start_token_id = input_tokenizer.cls_token_id
model.config.pad_token_id = input_tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

input_ids = input_tokenizer(["This is a really long text","so is this","so is this"], return_tensors="pt", padding=True).input_ids
labels = input_tokenizer(["This is a really long text","so is this","so is this"], return_tensors="pt", padding=True).input_ids
# outputs = model(input_ids=input_ids, labels=input_ids)
# loss, logits = outputs.loss, outputs.logits

# print(input_ids)
# print(loss)
# print(outputs.logits.shape)
# guess = torch.argmax(outputs.logits,dim=2).long()
# print(guess)
# print(output_tokenizer.decode(guess))



metric = evaluate.load("exact_match")
model.eval()
with torch.no_grad():
    outputs = model.generate(input_ids=input_ids)

pred = []
truth = []
for i in range(len(outputs)):
    pred.append(output_tokenizer.decode(outputs[i]))
    truth.append(input_tokenizer.decode(input_ids[i]))

print("======================")
print(pred)
print("======================")
print(truth)

metric.add_batch(predictions=pred, references=truth)
score = metric.compute()
print('Validation Accuracy:', score['exact_match'])