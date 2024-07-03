import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules.encoder_decoder import EncoderDecoder


class VQA_model(nn.Module):
    def __init__(self, args, tokenizer, text_extractor,backbone=None):
        super(VQA_model, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.text_extractor = text_extractor
        if not args.text_extractor == 'scratch':
            self.text_projection = nn.Parameter(torch.empty(self.text_extractor.transformer.width, args.d_model))
        
        self.fc = nn.Sequential(nn.LayerNorm(1723),nn.Linear(1723,args.d_model),nn.Linear(args.d_model,args.n_classes))

        if not backbone:
            print('use backbone: default')
            self.backbone = EncoderDecoder(args, tokenizer)
        else:
            print(f'use backbone: {args.caption_model}')
            self.backbone = backbone
            
        if args.dataset_name:
            self.forward = self.forward_brca
        else:
            raise ValueError('no forward function')
        self.freeze_extractor()

    def cal_parameters(self):
        # 定义总参数量、可训练参数量及非可训练参数量变量
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        # 遍历model.parameters()返回的全局参数列表
        for param in self.parameters():

            mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_brca(self, images, question_ids,question_masks,targets=None, target_masks=None,mode='train'):

        img_embeddings = images  # shape 1*N*384
        fc_feats = torch.sum(images,dim=1) #shape 1*384
       # question_ids, question_masks = question_ids[0], question_masks[0] # shape N*l
    
        truncate = int(torch.sum(question_masks))
        question_ids, question_masks = question_ids[:,:truncate], question_masks[:,:truncate]
        q_embeddings = self.text_extractor.encode_text(token_ids= question_ids,attention_mask= question_masks)
        if self.args.text_extractor =='scratch':

            q_embeddings[0,-1]= self.tokenizer.tokenizer_question.token2idx['<sep>']
            q_embeddings = self.backbone.model.tgt_embed(q_embeddings)
        else:
            q_embeddings = (q_embeddings @ self.text_projection)
        q_embeddings = q_embeddings[question_masks>0].unsqueeze(0)
        #embeddings = torch.cat([q_embeddings[question_masks>0].unsqueeze(0),img_embeddings],dim=1)

        
        if mode == 'train':
            output = self.backbone(img_embeddings, q_embeddings, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.backbone(att_feats = img_embeddings, q_feats = q_embeddings,mode='sample')
        elif mode == 'encode':
            output = self.backbone(embeddings, mode='encode')

            logits = self.fc(output[0,0,:]).unsqueeze(0)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return Y_hat, Y_prob
        else:
            raise ValueError
        return output

    def freeze_extractor(self):
        for param in self.text_extractor.parameters():
            param.requires_grad = False