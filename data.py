from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

def get_dataloader(tokenizer_name='gpt2', seq_length=512, batch_size=8, rank=0, world_size=1):
    """ Return the configured dataloader associated with the model. """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(data):
        return tokenizer(data["text"], truncation=True, max_length=seq_length, padding="max_length")

    def filter_empty_labels(example):
        return sum(1 for label in example['input_ids'] if label != tokenizer.pad_token_id) > 0

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]
    ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])
    ds = ds.filter(filter_empty_labels)
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(ds, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)

    return data_loader