import torch
import torch.nn as nn


class Joint_Representaion_Learner(nn.Module):
    def __init__(self, feats_size, opt):
        super(Joint_Representaion_Learner, self).__init__()
        self.fusion = opt.get('fusion', 'temporal_concat') # temporal concat, concat along time frame axis

        if self.fusion not in ['temporal_concat', 'addition', 'none']:
            raise ValueError('We now only support the fusion type: temporal_concat | addition | none')

        self.norm_list = []
        self.is_bn = (opt.get('norm_type', 'bn').lower() == 'bn')

        if not opt['no_encoder_bn']: # bagian ini inisialisasi layer BN aja
            if self.fusion == 'addition':
                feats_size = [feats_size[0]]
            for i, item in enumerate(feats_size): # feat_size is a list
                tmp_module = nn.BatchNorm1d(item) if self.is_bn else nn.LayerNorm(item)
                self.norm_list.append(tmp_module)
                self.add_module("%s%d"%('bn' if self.is_bn else 'ln', i), tmp_module)

    def forward(self, encoder_outputs, encoder_hiddens): # fungsi jalannya
        if not isinstance(encoder_hiddens, list):
            encoder_hiddens = [encoder_hiddens]
        encoder_hiddens = torch.stack(encoder_hiddens, dim=0).mean(0) # karena ada dua mean, jadi di-mean lagi

        if self.fusion == 'none': # pasti skip
            if isinstance(encoder_outputs, list):
                encoder_outputs = torch.cat(encoder_outputs, dim=1)
            return encoder_outputs, encoder_hiddens
        
        if not isinstance(encoder_outputs, list): # cek biar list dan bisa dirubah ke tensor
            encoder_outputs = [encoder_outputs]

        if self.fusion == 'addition': # skip aja
            encoder_outputs = torch.stack(encoder_outputs, dim=0).mean(0)

        if len(self.norm_list):
            assert len(encoder_outputs) == len(self.norm_list)
            for i in range(len(encoder_outputs)): # encoder_outputs memiliki panjang 2
                if self.is_bn: # ini pasti True
                    
                    # mengaplikasikan batch_norm ke setiap features
                    # dan melakukan reshape ke bentuk semula
                    if (i == 2):
                        batch_size, seq_len, num_objs, _ = encoder_outputs[i][1].shape # ambil r_hat
                        encoder_outputs[i] = (encoder_outputs[i][0],
                                              self.norm_list[i](encoder_outputs[i][1].contiguous()\
                                                                .view(batch_size * seq_len * num_objs, -1))\
                                                                    .view(batch_size, seq_len, num_objs, -1))
                    else:
                        batch_size, seq_len, _ = encoder_outputs[i].shape
                        encoder_outputs[i] = self.norm_list[i](encoder_outputs[i].contiguous().view(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)
                else:
                    encoder_outputs[i] = self.norm_list[i](encoder_outputs[i])

        if self.fusion == 'temporal_concat': # masuk ke sini
            assert isinstance(encoder_outputs, list)
            encoder_obj_outputs = encoder_outputs[-1]
            encoder_outputs = torch.cat(encoder_outputs[:2], dim=1) # concat sepanjang timeframe
        
        # shape normalized encoder_outputs [batch_size, seq_len, feature_dim]
        # shape averaged across timeframe encoder_hiddens [batch_size, 1, feature_dim] 
        return encoder_outputs, encoder_hiddens, encoder_obj_outputs
