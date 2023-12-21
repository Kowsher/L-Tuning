from torch import nn
from transformers import DataCollatorForLanguageModeling
import torch
import random

class DataCollatorForLTCls:
    def __init__(self, config, tokenizer, label2prompt):
        super().__init__()
        self.config = config
        tokenizer.padding_side = 'right'  # Set padding to the left

        extra_token = tokenizer.convert_tokens_to_ids('.')

        old_pad_id = tokenizer.pad_token_id
        tokenizer.pad_token_id=extra_token
        self.prompt_encodings = tokenizer(list(label2prompt.values()), padding=True, truncation=True, add_special_tokens=False)
        tokenizer.pad_token_id=old_pad_id
        #print('config.cls_token_id', config.cls_token_id, config.sep_token_id, len(self.prompt_encodings['input_ids']))
  
        if config.cls_token_id is not None and config.sep_token_id  is not None:
            for i in range(len(self.prompt_encodings['input_ids'])):
                self.prompt_encodings['input_ids'][i].insert(0, config.cls_token_id)
                self.prompt_encodings['input_ids'][i].append(config.sep_token_id)
        elif config.cls_token_id  is not None:
            for i in range(len(self.prompt_encodings['input_ids'])):
                self.prompt_encodings['input_ids'][i].insert(0, config.cls_token_id)
        elif config.sep_token_id  is not None:
            for i in range(len(self.prompt_encodings['input_ids'])):
                self.prompt_encodings['input_ids'][i].append(config.sep_token_id)
                
        print('Prefix length (pre_seq_len) is', len( self.prompt_encodings['input_ids'][0]))



        self.false_labels  = [[j for j in range(len(label2prompt)) if j != i] for i in range(len(label2prompt))]

        self.label2prompt = label2prompt


    def __call__(self, examples):



        first_half = len(examples) // 2 + 1
        first_half_list_of_example = [{**d, 'prefix_ids': self.prompt_encodings['input_ids'][d['label']], 'attention_mask_prefix': self.prompt_encodings['attention_mask'][d['label']], 'label': 1} for d in examples[0:first_half]]

        second_half_list_of_example = []


        for i in range(first_half, len(examples)):
            nb = random.choice( self.false_labels[examples[i]['label' ]])
            second_half_list_of_example.append({**examples[i], 'prefix_ids': self.prompt_encodings['input_ids'][nb], 'attention_mask_prefix': self.prompt_encodings['attention_mask'][nb], 'label': 0})

        results = first_half_list_of_example + second_half_list_of_example
        random.shuffle(results)
        #print(len(results))

        batch = {
            "input_ids": torch.tensor([item["input_ids"] for item in results]),
            "prefix_ids": torch.tensor([item["prefix_ids"] for item in results]),
            "attention_mask": torch.tensor([item["attention_mask"] for item in results]),
            "labels": torch.tensor([item["label"] for item in results]),


            # Add other keys as required...
        }
        #print(len(batch['input_ids']))

        return batch
