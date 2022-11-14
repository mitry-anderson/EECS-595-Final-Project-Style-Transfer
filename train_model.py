from transformers import GPT2Tokenizer, GPT2LMHeadModel, set_seed
import torch

set_seed(595)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
language_model = GPT2LMHeadModel.from_pretrained('gpt2')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
print(inputs)
outputs = language_model(**inputs, labels=inputs["input_ids"], output_hidden_states=True)
loss = outputs.loss
logits = outputs.logits
print(outputs.hidden_states[0].shape)

# print(outputs)
# print(loss)
print(logits.shape)
guess = torch.max(logits, dim=2).indices.squeeze()
print(guess)
print(tokenizer.decode(guess))


