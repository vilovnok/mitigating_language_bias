import torch
import torch.nn.functional as F
import torch.distributed as dist




def jt_module(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    input_ids = input_ids.view((-1, input_ids.size(-1)))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) 

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    batch_size = input_ids.size(0) // cls.data_args.num_sent
    pooler_output = pooler_output.view(batch_size, cls.data_args.num_sent, -1)

    z1, z2, z3 = pooler_output[:,0], pooler_output[:,1], pooler_output[:, 2]

    if dist.is_available() and cls.training:
        z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
        z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
        z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
        
        dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
        dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
        dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())

        z1_list[dist.get_rank()] = z1
        z2_list[dist.get_rank()] = z2
        z3_list[dist.get_rank()] = z3
        
        z1 = torch.cat(z1_list, 0)        
        z2 = torch.cat(z2_list, 0)        
        z3 = torch.cat(z3_list, 0)         

    # Подумать где нормализация должна быть
    z1 = F.normalize(z1, p=2, dim=-1)
    z2 = F.normalize(z2, p=2, dim=-1)
    z3 = F.normalize(z3, p=2, dim=-1)
    
    cos_sim = cls.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0))
    z1_z3_cos = cls.cosine_similarity(z1.unsqueeze(1), z3.unsqueeze(0))
    cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

    labels = torch.arange(cos_sim.size(0)).long().to(cls.device)

    z3_weight = cls.model_args.hard_negative_weight
    weights = torch.tensor(
            [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
        ).to(cls.device)
    cos_sim = cos_sim + weights
    loss = cls.loss_fun(cos_sim, labels)
    
    return loss



def jd_module(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels: torch.Tensor=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    input_ids = input_ids.view((-1, input_ids.size(-1)))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) 
    labels = labels.to(torch.float32)

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    batch_size = input_ids.size(0) // cls.data_args.jd_num_classes
    pooler_output = pooler_output.view(batch_size, cls.data_args.jd_num_classes, -1)

    premise, hypothesis = pooler_output[:, 0], pooler_output[:, 1]
    
    z = torch.cat((premise, 
                  hypothesis, 
                  torch.abs(hypothesis - premise), 
                  hypothesis * premise), dim=1)
    
    output = cls.jd_classifier(z)
    loss = cls.loss_fun_bce(output, labels)

    return loss



def jf_module(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels: torch.Tensor=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    input_ids = input_ids.view((-1, input_ids.size(-1)))
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) 
    labels = labels.to(torch.float32)

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True
    )

    z = cls.pooler(attention_mask, outputs)        
    output = cls.jf_classifier(z)
    loss = cls.loss_fun_ce(output, labels)
    
    return loss