import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
import random

template=(lambda x: f'what is {x} according to the slide?',
          lambda x: f'please predict {x}?',
          lambda x: f'Could you predict {x}?',
          lambda x: f'what is the result of {x} in the slide?',
          lambda x: f'From the slide, can you infer the {x}?',
          lambda x: f'Can you help the pathologist to predict {x}?')


    



class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir 
        self.ann_path = args.ann_path
        self.split_path = args.split_path
        
        self.max_seq_length = args.max_seq_length
        self.max_fea_length = args.max_fea_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.choice_counts = {'a': 0,'b': 0,'c': 0,'d': 0}
        cases = self.clean_data(pd.read_csv(self.split_path).loc[:, self.split].dropna())
        
        # REMOVE REPETITIVE SAMPLES
        if self.split in ('val','test'):
            tmp_cases = self.clean_data(pd.read_csv(self.split_path).loc[:, 'train'].dropna())
            for item in tmp_cases.keys():
                if item in cases:
                    del cases[item]

        
        self.examples = []
        
        with open(f'{self.ann_path}/WsiVQA_{split}.json', 'r', encoding='utf-8') as file:  
            data = json.load(file) 
            
        ids = [item['Id'] for item in data]

        count=0

        max_seq_length=0
        for dir in os.listdir(self.image_dir):
            if not 'DX1' in dir:
                continue
            
            image_path = os.path.join(self.image_dir,dir)

            pairs=[]
            case_id = dir[:12]
            for idx,name in enumerate(ids):
                if name==case_id:
                    pairs.append(data[idx])
                    assert data[idx]['Id']==case_id
            if len(pairs)>0:
                count+=1

            for item in pairs:
                
                question_ids=self.tokenizer.encode_input(item['Question'])
                answer_ids = self.tokenizer.encode_output(item['Answer'])
                choice =  item.get('Choice')
                
                question_ids, question_masks= self.pad_questions(question_ids)
                answer_ids, answer_masks, seq_length =self.pad_answers(answer_ids)
            
                if seq_length > max_seq_length:
                    max_seq_length = seq_length

                self.examples.append({'id':dir, 'image_path': image_path,'Questions': question_ids, 'Answers': answer_ids,
                                  'Question_Masks': question_masks, 'Answer_Masks': answer_masks,'split': self.split, 'Choices': choice})


        self.num_cases = count
        print(f'The size of {self.split} dataset: {count} sample with {len(self.examples)} vqa pairs')
        print(f'max seq length of answers {max_seq_length}')



    def __len__(self):
        return len(self.examples)
    





class BrcaImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        
        image = torch.load(image_path)
        image = image[:self.max_fea_length]
        question_ids = example['Questions']
        
        answer_ids = example['Answers']
        
        question_masks = example['Question_Masks']
        answer_masks = example['Answer_Masks']


        choice_ids = example['Choices']
        sample = (image_id, image, question_ids, question_masks,answer_ids,answer_masks,choice_ids)
        return sample
    
    def filter_vqa_choices(self,pairs_report):
        choices = ['a','b','c','d']
        
        choice2ans_map = {'a': lambda x:re.findall('a[:.)](.+?)(?:b[:.]|@)',x,flags=re.I),
         'b': lambda x:re.findall('b[:.)](.+?)(?:c[:.]|@)',x,flags=re.I),
         'c': lambda x:re.findall('c[:.)](.+?)(?:d[:.]|@)',x,flags=re.I),
         'd': lambda x:re.findall('d[:.)](.+?)@',x,flags=re.I)}
        pairs = []

        for item in pairs_report:
            q = item['Question']
            c = item['Choice'] +' @'
            a = item['Answer']
            
            if a in choices:
                answer = choice2ans_map[a](c)[0].strip()
                
            elif re.search('[abcd][.:)]',a):

                index = re.search('[abcd][.:)]',a).span()[0]
                a = a[index]
                
                if len(choice2ans_map[a](c))>1:
                    c =c.replace('.', ' ')
                elif len(choice2ans_map[a](c))==0:
                    return None
                answer = choice2ans_map[a](c)[0].strip()
                
            elif re.findall(' [abcd] ',' ' + a + ' '):
                #assert len(re.findall(' [abcd] ',' ' + a + ' '))==1
                a=re.findall(' [abcd] ',' ' + a + ' ')[0].strip()
                answer = choice2ans_map[a](c)[0].strip()
            elif not re.search(' [abcd] ',a): #contain only answer
                answer = a
            else:
                answer = None
                continue
                
            if a in choices:
                self.choice_counts[a]+=1
             
            #split choices
            choices_split = []
            for item in choices:
                if not len(choice2ans_map[item](c))==0:
                    choices_split.append(choice2ans_map[item](c)[0].strip())
           

            pairs.append({'Question':self.tokenizer.encode_input(q),
                          'Choice': choices_split,'Answer': self.tokenizer.encode_output(answer)})
        return pairs
    
    def generate_vqa_pairs(self, pairs):
        token_pairs = []
        for key,value in pairs.items():
            
            if not value:
                continue
            if key in ['days_to_last_folowup','days_to_death']:
                q = 'survival time'
            else:
                q = key
            
            index = random.randint(0,5) 
            question =  template[index](q) 
            answer = value
           
            token_pairs.append({'Question': self.tokenizer.encode_input(question), 'Answer': self.tokenizer.encode_output(answer)})

        return token_pairs
            
    def load_vqa_pairs(self,path):
        file_name = os.path.join(path, self.pairs_with_choice)
        file_name1 = os.path.join(path, self.pairs_no_choice)

        if not os.path.exists(file_name) or not os.path.exists(file_name1):
            print(f'file not exists: {path}')
            return None
        pairs_with_choice = json.loads(open(file_name, 'r').read())
        pairs_no_choice = json.loads(open(file_name1, 'r').read())

        tokens_with_choice = self.filter_vqa_choices(pairs_with_choice)
        tokens_no_choice = self.generate_vqa_pairs(pairs_no_choice)

        return pairs_with_choice, pairs_no_choice, tokens_with_choice, tokens_no_choice
    
    def clean_data(self,data):
        cases = {}
        for idx in range(len(data)):
            case_name = data[idx]

            case_id = '-'.join(case_name.split('-')[:3])
            cases[case_id] = case_name
        return cases 
    
    def filter_df(self,df, filter_dict):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            # assert 'label' not in filter_dict.keys()
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def df_prep(self,data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data
    
    
    def pad_questions(self,questions):

        if isinstance(questions[0],int):
            seq_len = len(questions)
            mask = [1]*seq_len
            return questions,mask
        else:
            max_seq_length = max([len(item) for item in questions])
            masks = []
            for i in range(len(questions)):
                item = questions[i]
                mask = [1]*len(item)
                padding = [0] * (max_seq_length-len(item))
                item.extend(padding)
                mask.extend(padding)
                masks.append(mask)
            return questions,masks
    
    def pad_answers(self,answers):
        if isinstance(answers[0],int):
            seq_len = len(answers)
            mask = [1]*seq_len
            return answers,mask,seq_len
        else:
            max_seq_length = max([len(item) for item in answers])
            masks = []
            for i in range(len(answers)):
                item = answers[i]
                mask = [1]*len(item)
                padding = [0] * (max_seq_length-len(item))
                item.extend(padding)
                mask.extend(padding)
                masks.append(mask)

        return answers,masks,max_seq_length
