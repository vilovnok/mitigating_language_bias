import torch

from transformers import PreTrainedTokenizer




def jt_prepare_features(examples, column_names: list, tokenizer: PreTrainedTokenizer):
    sent0_name = column_names[0]
    sent1_name = column_names[1]
    sent2_name = column_names[2]
    total = len(examples[sent0_name])

    for idx in range(total):
        if examples[sent0_name][idx] is None:
            examples[sent0_name][idx] = " "
        if examples[sent1_name][idx] is None:
            examples[sent1_name][idx] = " "
    
    sentences = examples[sent0_name] + examples[sent1_name]
    if sent2_name is not None:
        for idx in range(total):
            if examples[sent2_name][idx] is None:
                examples[sent2_name][idx] = " "
        sentences += examples[sent2_name]
        
    sent_features = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=64 # TODO: заменить на dynamic value
    )

    features = {}
    if sent2_name is not None:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
    else:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
    
    return features


def jd_prepare_features(examples, column_names: list, num_classes: int, tokenizer: PreTrainedTokenizer):
    sent0_name = column_names[0]
    sent1_name = column_names[1]
    label_name = column_names[3]
    total = len(examples[sent0_name])

    for idx in range(total):
        if examples[sent0_name][idx] is None:
            examples[sent0_name][idx] = " "
        if examples[sent1_name][idx] is None:
            examples[sent1_name][idx] = " "
    
    sentences = examples[sent0_name] + examples[sent1_name]
    sent_features = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=64 # TODO: заменить на dynamic value
    )

    features = {}
    for key in sent_features:
        features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]

    labels = examples[label_name]
    multilabels = []
    for l in labels:
        one_hot = [0]*num_classes
        if l < num_classes:
            one_hot[l] = 1
        multilabels.append(one_hot)
    
    features[label_name] = torch.tensor(multilabels)
    return features


def jf_prepare_features(examples, column_names: list, num_classes: int, tokenizer: PreTrainedTokenizer):
    sent0_name = column_names[1]
    label_name = column_names[3]
    total = len(examples[sent0_name])

    for idx in range(total):
        if examples[sent0_name][idx] is None:
            examples[sent0_name][idx] = " "
    
    sent_features = tokenizer(
        examples[sent0_name],
        padding="max_length",
        truncation=True,
        max_length=512  # TODO: заменить на dynamic value
    )

    features = {}
    for key in sent_features:
        features[key] = [sent_features[key][i] for i in range(total)]

    labels = examples[label_name]
    multilabels = []
    for l in labels:
        one_hot = [0]*num_classes
        if l < num_classes:
            one_hot[l] = 1
        multilabels.append(one_hot)
    
    features[label_name] = multilabels
    return features
