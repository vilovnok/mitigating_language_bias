import torch
import torch.nn as nn




class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)


class Similarity(nn.Module):
    def __init__(self, temp):
        super(Similarity, self).__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    def __init__(self, pooler_type):
        super(Pooler, self).__init__()
        self.pooler_type = pooler_type

        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


def prepare_features(examples, tokenizer, column_names: list):
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
        max_length=64
    )

    features = {}
    if sent2_name is not None:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
    else:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
    return features
