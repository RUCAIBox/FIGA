import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import Levenshtein
import torch.nn.functional as F

import torch
import transformers
import json
from accelerate.utils import set_seed
from datasets import load_dataset

set_seed(42)

from torch.utils.data import Dataset, DataLoader
from transformers import Trainer

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "no_instruction_input": ("{instruction}\n\n{input}\n"),
    "no_instruction_no_input": ("{instruction}\n\n"),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    instruction_type: str = field(default='default')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    ids_list = tokenizer(
        strings,
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_attention_mask=False
    )['input_ids']

    input_ids = []
    input_ids_lens = []

    for ids in ids_list:
        input_ids.append(torch.tensor(ids))
        input_ids_lens.append(len(ids))

    return dict(
        input_ids=input_ids,
        input_ids_lens=input_ids_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    raws: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + ' ' + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
        
    examples2 = [s + ' ' + t for s, t in zip(sources, raws)]
    examples_tokenized2, sources_tokenized2 = [_tokenize_fn(strings, tokenizer) for strings in (examples2, sources)]
    input_ids2 = examples_tokenized2["input_ids"]
    raw_labels = copy.deepcopy(input_ids2)
    for label, source_len in zip(raw_labels, sources_tokenized2["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, raw_input_ids=input_ids2, labels=labels, raw_labels=raw_labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, instruction_type: str):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_dataset(data_path)['train']

        logging.warning("Formatting inputs...")
        if instruction_type == 'no_inst':
            prompt_input, prompt_no_input = PROMPT_DICT["no_instruction_input"], PROMPT_DICT["no_instruction_no_input"]
        else:
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['revised_output']}{tokenizer.eos_token}" for example in list_data_dict]
        raw = [f"{example['original_output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, raw, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.raw_input_ids = data_dict["raw_input_ids"]
        self.labels = data_dict["labels"]
        self.raw_labels = data_dict["raw_labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], raw_input_ids=self.raw_input_ids[i], labels=[self.labels[i], self.raw_labels[i]])

def transform_tensor(source, target):
    operations = Levenshtein.editops(source.tolist(), target.tolist())
    return [i[2] for i in operations if i[0]!='delete'], [i[1] for i in operations if i[0]!='insert']

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        raw_input_ids = [instance["raw_input_ids"] for instance in instances]
        labels = [instance["labels"][0] for instance in instances]
        raw_labels = [instance["labels"][1] for instance in instances]

        max_len = max(max(len(seq) for seq in input_ids), max(len(seq) for seq in raw_input_ids))
        input_ids = [F.pad(seq, (0, max_len - len(seq)), value=self.tokenizer.pad_token_id) for seq in input_ids]
        raw_input_ids = [F.pad(seq, (0, max_len - len(seq)), value=self.tokenizer.pad_token_id) for seq in raw_input_ids]
        input_ids = torch.stack(input_ids)
        raw_input_ids = torch.stack(raw_input_ids)

        max_len_labels = max(max(len(seq) for seq in labels), max(len(seq) for seq in raw_labels))
        my_mask = torch.zeros(input_ids.shape[0], max_len_labels)
        my_mask_neg = torch.zeros(input_ids.shape[0], max_len_labels)

        # Positive masks
        for idx, exmp in enumerate(labels):
            operations, op=transform_tensor(raw_labels[idx], labels[idx])
            for j in operations:
                my_mask[idx][j]=1
            for j, _ in enumerate(labels[idx]):
                if labels[idx][j]==self.tokenizer.eos_token_id:
                    my_mask[idx][j]=1
                    break

        # Negative masks
        for idx, exmp in enumerate(raw_labels):
            op, operations = transform_tensor(raw_labels[idx], labels[idx])
            for j in operations:
                my_mask_neg[idx][j] = -1
            for j, _ in enumerate(raw_labels[idx]):
                if raw_labels[idx][j]==self.tokenizer.eos_token_id:
                    my_mask_neg[idx][j]=1
                    break
        labels = [F.pad(seq, (0, max_len_labels - len(seq)), value=IGNORE_INDEX) for seq in labels]
        raw_labels = [F.pad(seq, (0, max_len_labels - len(seq)), value=IGNORE_INDEX) for seq in raw_labels]
        labels = torch.stack(labels)
        raw_labels = torch.stack(raw_labels)
  
        all_input_ids = torch.stack([torch.cat([input_ids[i], raw_input_ids[i]]) for i in range(len(input_ids))]).view(-1, input_ids.size(1))
        all_labels = torch.stack([torch.cat([labels[i], raw_labels[i]]) for i in range(len(labels))]).view(-1, labels.size(1))
        all_attention_mask = torch.stack([torch.cat([input_ids.ne(self.tokenizer.pad_token_id)[i], raw_input_ids.ne(self.tokenizer.pad_token_id)[i]]) for i in range(len(input_ids))]).view(-1, input_ids.size(1))
        all_my_mask = torch.stack([torch.cat([my_mask[i], my_mask_neg[i]]) for i in range(len(my_mask))]).view(-1, my_mask.size(1))

        return dict(
            input_ids=all_input_ids,
            labels=all_labels,
            attention_mask=all_attention_mask,
            my_mask=all_my_mask,
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, instruction_type=data_args.instruction_type)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
