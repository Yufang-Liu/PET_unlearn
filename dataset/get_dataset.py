from torch.utils.data import DataLoader, sampler, Subset, ConcatDataset
from datasets import load_dataset
import random


def get_dataset(dataset_str, config, tokenizer):
    dataset_path = config['data'][dataset_str + '_path']
    dataset_name = config['data'][dataset_str + '_dataset_name']
    dataset_describe = config['data'][dataset_str + '_describe']
    dataset_column = config['data'][dataset_str + '_column_name']
    batch_size = config['trainer']['batch_size']

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples[dataset_describe], truncation=True,
                            max_length=config['model']['max_len'])
        return outputs

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    datasets = load_dataset(dataset_name, cache_dir=dataset_path)
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset_column,
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels
    # by the models of the transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels") \
        if dataset_str != 'yahoo' \
        else tokenized_datasets.rename_column("topic", "labels")

    if config['options']['sample_small']:
        n_train = len(tokenized_datasets["train"])
        train_split = n_train // 10
        train_indices = list(range(n_train))
        random.shuffle(train_indices)
        train_sampler = sampler.SubsetRandomSampler(train_indices[:train_split])

        n_test = len(tokenized_datasets["test"])
        test_split = n_test // 10
        test_indices = list(range(n_test))
        random.shuffle(test_indices)
        test_sampler = sampler.SubsetRandomSampler(test_indices[:test_split])

        train_dataloader = DataLoader(tokenized_datasets["train"], collate_fn=collate_fn,
                                      batch_size=batch_size, sampler=train_sampler)
        eval_dataloader = DataLoader(
            tokenized_datasets["test"], collate_fn=collate_fn, batch_size=batch_size,
            sampler=test_sampler
        )
        print("load samll dataset, number of batch: {}".format(len(train_dataloader)))
    else:
        train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn,
                                      batch_size=batch_size)
        eval_dataloader = DataLoader(
            tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
        )
        print("load full dataset, number of batch: {}".format(len(train_dataloader)))
    return train_dataloader, eval_dataloader


def get_all_dataset(config, tokenizer):
    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples[dataset_describe], truncation=True,
                            max_length=config['model']['max_len'])
        return outputs

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    task_list = config['data']['multi_task']
    batch_size = config['trainer']['batch_size']
    train_list, test_list = [], []
    for task_id, dataset_str in enumerate(task_list):
        dataset_path = config['data'][dataset_str + '_path']
        dataset_name = config['data'][dataset_str + '_dataset_name']
        dataset_describe = config['data'][dataset_str + '_describe']
        dataset_column = config['data'][dataset_str + '_column_name']

        # datasets = load_dataset(dataset_name, cache_dir=dataset_path)
        # train_set, test_set = datasets['train'], datasets['test']
        train_set = load_dataset(dataset_name, cache_dir=dataset_path,
                                 split='train[:10%]')
        test_set = load_dataset(dataset_name, cache_dir=dataset_path,
                                split='test[:10%]')

        train_set = train_set.add_column(name="task_id", column=[task_id]*len(train_set))
        test_set = test_set.add_column(name="task_id", column=[task_id]*len(test_set))

        tokenized_train_datasets = train_set.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset_column,
        )
        tokenized_test_datasets = test_set.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset_column,
        )


        # We also rename the 'label' column to 'labels' which is the expected name for labels
        # by the models of the transformers library
        tokenized_train_datasets = tokenized_train_datasets.rename_column("label", "labels") \
            if dataset_str != 'yahoo' \
            else tokenized_train_datasets.rename_column("topic", "labels")

        tokenized_test_datasets = tokenized_test_datasets.rename_column("label", "labels") \
            if dataset_str != 'yahoo' \
            else tokenized_test_datasets.rename_column("topic", "labels")

        train_list.append(tokenized_train_datasets)
        test_list.append(tokenized_test_datasets)

    full_trainset = ConcatDataset(train_list)
    full_testset = ConcatDataset(test_list)

    train_dataloader = DataLoader(
        full_trainset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    eval_dataloader = DataLoader(
        full_testset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    return train_dataloader, eval_dataloader

