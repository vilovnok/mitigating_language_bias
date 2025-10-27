from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead

from src.utils import Similarity, Pooler
from src.module import jt_module, jd_module, jf_module




class JobMatchingMultiTaskModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.model_args = kwargs["model_args"]
        self.data_args = kwargs["data_args"]
        self.pooler_type = kwargs["model_args"].pooler_type
        self.pooler = Pooler(kwargs["model_args"].pooler_type)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cosine_similarity = Similarity(temp=self.model_args.temp)
        self.loss_fun_ce = nn.CrossEntropyLoss()
        self.loss_fun_bce = nn.BCEWithLogitsLoss()

        self.jd_classifier = MLP(
            input_dim= 4 * kwargs["model_args"].model_hidden_dim, 
            hidden_dim=kwargs["model_args"].mlp_hidden_dim,
            num_classes=kwargs["data_args"].jd_num_classes
            
        )        
        self.jf_classifier = MLP(
            input_dim= kwargs["model_args"].model_hidden_dim, 
            hidden_dim=kwargs["model_args"].mlp_hidden_dim,
            num_classes=kwargs["data_args"].jf_num_classes
        )        
        
        self.init_weights()


    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        jt_labels=None,
        jd_labels=None,
        jf_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        total_loss = 0.0
        total_loss_metadata = {}
    
        jt_loss = jt_module(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=jt_labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )    
        total_loss += jt_loss     


        jd_loss = jd_module(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=jd_labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels
            )  
        total_loss += jd_loss
        
        
        jf_loss = jf_module(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=jf_labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels
            )  
        total_loss += jf_loss
        
        total_loss_metadata['jt_loss'] = jt_loss
        total_loss_metadata['jd_loss'] = jd_loss
        total_loss_metadata['jf_loss'] = jf_loss
        total_loss_metadata['total_loss'] = total_loss
        return total_loss_metadata
