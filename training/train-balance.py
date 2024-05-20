# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import copy
IGNORE_INDEX = -100

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
torch.set_printoptions(profile="full")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    topk : int = field(default=1)
    expert_nums : int = field(default=7)
    base_set: str = field(default="")
    router_aux_loss_coef : float =field(default=0.1)



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    conv_template: str = field(
        default=None, metadata={"help": "Template used to format the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    source_model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Original maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Expanded maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    only_train_gate: bool = False
    pretrain_loss: bool = False


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def get_prompt(query):
    prompt=f"<s>[INST]\n{query}\n[/INST]"
    return prompt


def preprocess_data(
    raw_data,
) -> Dict:
    
    new_data = []
    for data in raw_data:
        question = "<s>" + data["messages"][0]['content']
        answers = data["messages"][1]['content']
        query = get_prompt(question)
        
        dic = {}
        dic["prompt"] = query
        dic["answer"] = answers
        dic["type"] = "train"
        
        new_data.append(dic)
    return new_data

def preprocess_data1(
    raw_data,
) -> Dict:
    
    new_data = []
    for data in raw_data:
        question = data["messages"][0]['content']
        answers = data["messages"][1]['content']
        dic = {}
        dic["prompt"] = question
        dic["answer"] = answers
        dic["type"] = "eval"
        new_data.append(dic)
        
    return new_data

def preprocess(
    sources,
    targets,
    tokenizer,
    pretrain_loss
):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    if not pretrain_loss:
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
    
    return dict(input_ids=input_ids, labels=labels)


def preprocess1(
    sources,
    targets,
    tokenizer,
    pretrain_loss
):
    """Preprocess the data by tokenizing."""
            
    encoded_inputs = tokenizer.batch_encode_plus(
        sources,
        padding="longest",
        truncation="longest_first",
        return_tensors='pt'
    )
    encoded_labels = tokenizer.batch_encode_plus(
        targets,
        padding="longest",
        truncation="longest_first",
        return_tensors='pt'
    )
    
    input_ids = encoded_inputs.input_ids
    labels = encoded_labels.input_ids
    print("174",input_ids.shape,labels.shape)
    
    return dict(input_ids=input_ids, labels=labels)

class Collator(object):

    def __init__(self,tokenizer,pretrain_loss):
        self.tokenizer = tokenizer
        self.pretrain_loss = pretrain_loss
        pass

    def __call__(self, batch):
        sources = []
        targets = []
        for instance in batch:
            source = instance['input_ids']
            target = instance['labels'][0]
            sources.append(source)
            targets.append(target)
        
        if batch[0]['labels'][1] == "train":
            data_dict = preprocess(sources, targets, self.tokenizer, self.pretrain_loss)
        else:
            data_dict = preprocess1(sources, targets, self.tokenizer, False)
            
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

class LazySupervisedDataset1(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_length=8192,data_type='train'):
        super(LazySupervisedDataset1, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.max_length = max_length
        self.data = preprocess_data(raw_data) if data_type == "train" else preprocess_data1(raw_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        query = self.data[i]["prompt"]
        answer = self.data[i]["answer"] + "</s>"
        
        return dict(
            input_ids=query,
            labels=(answer,self.data[i]['type']),
        )

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)



def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args,max_length,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset1
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path,'r'))
    if data_args.eval_data_path is not None:
        train_raw_data = raw_data
        eval_raw_data = json.load(open(data_args.eval_data_path,'r'))

    else:
        # Split train/test
        perm = np.random.permutation(len(raw_data))
        split = int(len(perm) * 0.98)
        train_indices = perm[:split]
        eval_indices = perm[split:]
        train_raw_data = [raw_data[i] for i in train_indices]
        eval_raw_data = [raw_data[i] for i in eval_indices]

    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")
    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, max_length=max_length,data_type="train")
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer , max_length=max_length,data_type="eval")
    print(len(eval_dataset))
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,max_length=training_args.model_max_length)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
    config._flash_attn_2_enabled=True
    config.topk = model_args.topk
    config.output_router_logits = True
    config.router_aux_loss_coef = model_args.router_aux_loss_coef
    config.base_set = json.loads(model_args.base_set)
    config.expert_nums = model_args.expert_nums


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        use_safetensors=False,
    )

    if training_args.only_train_gate:
        for n,p in model.named_parameters():
            p.requires_grad_(False)
            
        for n,p in model.named_parameters():
            if "self_attn.gate" in n:
                p.requires_grad_(True)
                print(n)
        model.enable_input_require_grads() 


    
    print(model)
    
    model.config.use_cache = False
    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer,
        args=training_args,
        data_collator = Collator(tokenizer,training_args.pretrain_loss),
        **data_module
    )

    
    trainer.train()
        
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
