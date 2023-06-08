import torch
import torch.nn as nn
from config import Constants
from models.bert import BertLayerNorm


class Seq2Seq(nn.Module): 
    def __init__(self, 
                 opt,
                 preEncoder=None, # Default None
                 encoder=None,
                 joint_representation_learner=None,
                 auxiliary_task_predictor=None, # Default None
                 decoder=None,
                 tgt_word_prj=None,
                 **kwargs
                 ):
        super(Seq2Seq, self).__init__()
        self.opt = opt
        self.preEncoder = preEncoder
        self.encoder = encoder
        self.joint_representation_learner = joint_representation_learner
        self.auxiliary_task_predictor = auxiliary_task_predictor
        self.decoder = decoder
        self.tgt_word_prj = tgt_word_prj

        if opt.get('tie_weights', False):
            self._tie_weights(opt['vocab_size'])

    def _tie_weights(self, vocab_size):
        word_embeddings = self.decoder.get_word_embeddings()
        self.tgt_word_prj.weight = word_embeddings.weight
        self.tgt_word_prj.bias = nn.Parameter(torch.zeros(vocab_size).float(), requires_grad=True)

    def encode(self, feats, **kwargs):
        results = {}
        if self.opt.get('automatic_mask', False): # this will be skipped
            attention_mask = []
            for feat in feats:
                assert len(feat.shape) == 3
                attention_mask.append(feat.sum(-1).eq(0))
            results['attention_mask'] = attention_mask

        if self.preEncoder is not None: # this condition will be skipped too
            feats = self.preEncoder(input_feats=feats)

        enc_output, enc_hidden, *attentions = self.encoder(feats)

        if len(attentions): # this condition will be skipped too
            results['encoder_attentions'] = attentions[0]

        if self.joint_representation_learner is not None:
            enc_output, enc_hidden, enc_obj_output = self.joint_representation_learner(enc_output, enc_hidden)

        if self.auxiliary_task_predictor is not None:
            auxiliary_results = self.auxiliary_task_predictor(
                enc_output=enc_output, 
            )
            results.update(auxiliary_results)

        results['enc_output'] = enc_output # ini tuh udah tensor yang disimpan di dict
        results['enc_hidden'] = enc_hidden # ini juga udah tensor
        results['enc_obj_output'] = enc_obj_output         
        return results

    def align_object_variable(self, r_feats, r_hat):
        '''
        align object modul according to ORG-TRL Paper
        refers = https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Object_Relational_Graph_With_Teacher-Recommended_Learning_for_Video_Captioning_CVPR_2020_paper.pdf
        args:
        object_variable : This is object features exctracted from Faster RCNN
        output:
        aligned_object_variable
        '''
        ## Mengambil anchor frame sebagai acuan untuk setiap objek
        ## Memisahkan anchor frame dari keseluruhan fitur objek
        anchor_frame = r_feats[:, 0]
        next_frame = r_feats[:, 1:r_feats.size(1)]
        
        ## menghitung cosine similarity scores
        ## matmul( achor_frame, next_frame ) / | anchor_frame | * | next_frame |
        similarity_score = (torch.matmul(anchor_frame.unsqueeze(1), next_frame.transpose(2, -1)) / \
                            (torch.norm(anchor_frame.unsqueeze(1), dim=-1)[:, :, :, None] * \
                            torch.norm(next_frame, dim=-1)[:, :, None, :]))

        aligned_frames = torch.gather(r_hat[:, 1:r_hat.size(1)], 
                                      dim=2, 
                                      index=similarity_score.topk(1, -1)[1].\
                                      expand(-1, -1, -1, r_hat.size(-1)))
        
        return torch.cat([r_hat[:, 0].unsqueeze(1), aligned_frames], dim=1)
    
    def prepare_inputs_for_decoder(self, encoder_outputs, category):
        input_keys_for_decoder = ['enc_output', 'enc_obj_output']
        if self.opt['decoding_type'] == 'LSTM': # this is skipped
            input_keys_for_decoder.append('enc_hidden')

        if self.opt.get('attribute', False) and self.opt.get('attribute_mode', 'none') != 'none':
            input_keys_for_decoder += ['attr_probs', 'video2attr_raw_scores']
        
        inputs_for_decoder = {'category': category}
        for key in input_keys_for_decoder:
            inputs_for_decoder[key] = encoder_outputs[key] # di sini udah ambil tensor
        
        if isinstance(inputs_for_decoder['enc_output'], list):
            assert len(inputs_for_decoder['enc_output']) == 1
            inputs_for_decoder['enc_output'] = inputs_for_decoder['enc_output'][0]
        
        if isinstance(inputs_for_decoder['enc_obj_output'], tuple):
            assert len(inputs_for_decoder['enc_obj_output']) == 2 # memeriksa objek fitur
            b_size, f_len, n_obj, _ = inputs_for_decoder['enc_obj_output'][1].size()
            inputs_for_decoder['enc_obj_output'] = self.align_object_variable(inputs_for_decoder['enc_obj_output'][0],
                                                                              inputs_for_decoder['enc_obj_output'][1]).view(b_size, f_len * n_obj, -1)

        return inputs_for_decoder

    def forward(self, **kwargs):
        func_name = "forward_" + self.opt['decoding_type']
        return getattr(self, func_name, None)(kwargs)

    def forward_NARFormer(self, kwargs):
        feats, tgt_tokens, category = map(
            lambda x: kwargs.get(x, None),
            ["feats", "tgt_tokens", "category"]
        )

        results = self.encode(feats)
        inputs_for_decoder = self.prepare_inputs_for_decoder(results, category)
        hidden_states, embs, *_ = self.decoder(
            tgt_seq=tgt_tokens, 
            **inputs_for_decoder
        )

        if not isinstance(hidden_states, list):
            hidden_states = [hidden_states]

        tgt_word_logits = [self.tgt_word_prj(item) for item in hidden_states]
        tgt_word_logprobs = [torch.log_softmax(item, dim=-1) for item in tgt_word_logits]

        results.update({
            Constants.mapping['lang'][0]: tgt_word_logprobs,
        })
        return results

    def forward_ARFormer(self, kwargs):
        feats, tgt_tokens, category = map(
            lambda x: kwargs.get(x, None),
            ["feats", "tgt_tokens", "category"]
        )
        decoding_type = kwargs.get('decoding_type', self.opt['decoding_type']) # ARFormr | NARFormer
        pmlm_flag = (decoding_type == 'SelfMask')
        if pmlm_flag:
            tgt_tokens = [item[:, 1:] for item in tgt_tokens] if isinstance(tgt_tokens, list) else tgt_tokens[:, 1:]
        else:
            tgt_tokens = [item[:, :-1] for item in tgt_tokens] if isinstance(tgt_tokens, list) else tgt_tokens[:, :-1]

        results = self.encode(feats)
        inputs_for_decoder = self.prepare_inputs_for_decoder(results, category) # inputs_for_decoder['enc_output'] : tensor(feats)
        hidden_states, embs, *_ = self.decoder( # hidden_states is equal to [ layer_output ]
            tgt_seq=tgt_tokens, 
            decoding_type=decoding_type,
            output_attentions=kwargs.get('output_attentions', False),
            **inputs_for_decoder # difeed setiap key value pairs ke forward method decoder
            )

        if not isinstance(hidden_states, list):
            hidden_states = [hidden_states]

        tgt_word_logits = [self.tgt_word_prj(item) for item in hidden_states]
        tgt_word_logprobs = [torch.log_softmax(item, dim=-1) for item in tgt_word_logits]
        
        results.update({
            Constants.mapping['lang'][0]: tgt_word_logprobs,
        })
        return results
