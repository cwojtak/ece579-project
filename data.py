def get_datasets():
    texts = []
    labels = []
    eval_texts = []
    eval_labels = []
    i = 0

    with open("data/SMSSpamCollection", "r") as data_file:
        data_file_contents = data_file.readlines()
        for line in data_file_contents:
            divide = line.find("am") + 2
            label = 1 if line[:divide] == "spam" else 0
            text = line[divide + 1:]
            if i % 2 == 0:
                texts.append(text)
                labels.append(label)
            else:
                eval_texts.append(text)
                eval_labels.append(label)
            i += 1
    return texts, labels, eval_texts, eval_labels
