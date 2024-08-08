import json
import re
from collections import Counter
import os
from transformers import AutoTokenizer, BertTokenizerFast

class Tokenizer(object):
    def __init__(self, args):
        self.args=args
        self.ann_path = args.ann_path
        self.threshold = args.threshold

        #self.dataset_name = args.dataset_name
        self.dataset_name = 'BRCA'
        if self.dataset_name == 'BRCA':
            self.clean_report = self.clean_report_brca

       
        self.token2idx, self.idx2token = self.create_vocabulary()
        self.vocab_size = self.get_vocab_size()
        assert args.bos_idx == self.token2idx['<bos>'], self.token2idx['<bos>']
        print(f"id of BOS:{self.token2idx['<bos>']}")
        
        
    def create_vocabulary(self):
        total_tokens = []

        tmp=[]
        for js in os.listdir(self.ann_path):
            with open(os.path.join(self.ann_path,js), 'r', encoding='utf-8') as file:    
                data = json.load(file) 

            for item in data:
                del item['Id']
            tokens = self.clean_report(str(data)).split()
            tmp.extend(tokens)



        counter = Counter(tmp)
        if self.args.text_extractor =='scratch':
            vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>'] +['<bos>'] + ['<sep>']

        else:
            vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>'] +['<bos>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token

        return token2idx, idx2token

    def clean_report_brca(self, report):

        report_cleaner = lambda t: (t.replace('\n', ' ').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('?',' ')\
            .strip().lower() + ' ').split('. ')
        sent_cleaner = lambda t: re.sub('[#,?;*!^&_+():-\[\]{}]', ' ', t.replace('"', '').
                                    replace('\\', '').replace("'", '').strip().lower()).replace('  ', ' ')

        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) 

        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids =[self.token2idx['<bos>']] + ids + [0]
        return {'input_ids': ids}

    def decode(self, ids):
        txt = self.idx2token[ids]
        #for i, idx in enumerate(ids):
        #    if idx > 0:
         #       if i >= 1:
         #           txt += ' '
          #      txt += self.idx2token[idx]
         #   else:
          #      break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
    
    
class MixedTokenizer(object):
    def __init__(self, tokenizer,tokenizer_a):
        self.tokenizer_question = tokenizer
        self.tokenizer_answer = tokenizer_a
        self.vocab_size = self.tokenizer_answer.vocab_size
        
        
    def encode_input(self, report):
        return self.tokenizer_question(report)['input_ids']
    
    def encode_output(self,report):
        return self.tokenizer_answer(report)['input_ids']

    
    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            idx = int(idx)
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.tokenizer_answer.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
    
    def decode_input(self, ids):
        txt = ''

        for i, idx in enumerate(ids):
            idx = int(idx)
            if idx > 0:
                if i >= 1:
                    txt += ' '

                txt += self.tokenizer_question.decode(idx)
            else:
                break
        return txt
    
    def decode_inputs(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode_input(ids))
        return out

def Build_Tokenizer(args):
    if args.text_extractor == 'bioclinicalbert':
        model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        tokenizer = BertTokenizerFast.from_pretrained('/chenpingyi/projects/WSI-GPT/WsiVQA/src/bioclinicalbert', tokenizer_class=AutoTokenizer)
        
    elif args.text_extractor == 'pubmedbert':
        model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        tokenizer = BertTokenizerFast.from_pretrained('/chenpingyi/projects/WSI-GPT/WsiVQA/src/pubmedbert', tokenizer_class=AutoTokenizer)

    elif args.text_extractor == 'llama':
        from transformers import LlamaForCausalLM, LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained("/output/path")
    else:
        tokenizer = Tokenizer(args)
        
    tokenizer_a = Tokenizer(args)
    return MixedTokenizer(tokenizer,tokenizer_a)
        
