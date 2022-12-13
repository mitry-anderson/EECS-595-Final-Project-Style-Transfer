from nltk.corpus import brown

# get the category names
categories = brown.categories()

# 80% for training, 18% for validation, rest (2%), to nearest integer for test.
train_percent = 0.8
val_percent = 0.18

#  for each category, load data and create 3 files (train, val, test)
for category in categories:
    # get data
    sentences = brown.sents(categories=[category])

    # join into a space separated line
    i = 0
    paragraphs_list = []
    paragraph = ''
    for sentence in list(sentences):
        paragraph += ' '.join(sentence)
        i += 1
        if i == 2:
            paragraphs_list.append(paragraph)
            paragraph = ''
            i = 0

    # calculate indices
    num_sentences = len(paragraphs_list)
    num_train = int(train_percent*num_sentences)
    num_val = int(val_percent*num_sentences)

    # separate into 3 datasets
    train_sentences = paragraphs_list[:num_train]
    val_sentences = paragraphs_list[num_train:(num_train + num_val)]
    test_sentences = paragraphs_list[(num_train + num_val):]

    # write dataset files
    with open('dataset/' + category + '_train.txt','w') as file:
        file.write('\n'.join(train_sentences))
    with open('dataset/' + category + '_validation.txt','w') as file:
        file.write('\n'.join(val_sentences))
    with open('dataset/' + category + '_test.txt','w') as file:
        file.write('\n'.join(test_sentences))

    # print a nifty little message
    print(f"{category}: {num_sentences} total paragraphs, {len(train_sentences)} training, {len(val_sentences)} validation, {len(test_sentences)} test")

















