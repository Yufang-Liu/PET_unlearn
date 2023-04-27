from torch.utils.data import DataLoader, sampler, Subset, ConcatDataset
from datasets import load_dataset, load_from_disk
from split_dataset import get_indices


def get_dataset(dataset_str, config, tokenizer, add_task_id=False):
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

    if not config['options']['sample_small']:
        train_set = load_dataset(dataset_name, cache_dir=dataset_path,
                                 split='train')
        test_set = load_dataset(dataset_name, cache_dir=dataset_path,
                                split='test')
    else:
        dataset = load_from_disk(dataset_path + '/small_' + dataset_name)
        train_set, test_set = dataset['train'], dataset['test']
        print(len(train_set), len(test_set))

    if add_task_id:
        task_id = config['data']['multi_task'].index(dataset_str)
        train_set = train_set.add_column(name="task_id", column=[task_id] * len(train_set))
        test_set = test_set.add_column(name="task_id", column=[task_id] * len(test_set))


    train_tokenized_datasets = train_set.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset_column,
    )

    test_tokenized_datasets = test_set.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset_column,
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels
    # by the models of the transformers library
    train_tokenized_datasets = train_tokenized_datasets.rename_column("label", "labels") \
        if dataset_str != 'yahoo' \
        else train_tokenized_datasets.rename_column("topic", "labels")

    test_tokenized_datasets = test_tokenized_datasets.rename_column("label", "labels") \
        if dataset_str != 'yahoo' \
        else test_tokenized_datasets.rename_column("topic", "labels")

    train_dataloader = DataLoader(train_tokenized_datasets, shuffle=True, collate_fn=collate_fn,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(
        test_tokenized_datasets, shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )
    print("load {} dataset, number of batch: {}".
          format("small" if config['options']['sample_small'] else "full", len(train_dataloader)))
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

        if not config['options']['sample_small']:
            train_set = load_dataset(dataset_name, cache_dir=dataset_path,
                                     split='train')
            test_set = load_dataset(dataset_name, cache_dir=dataset_path,
                                    split='test')
        else:
            dataset = load_from_disk(dataset_path + '/small_' + dataset_name)
            train_set, test_set = dataset['train'], dataset['test']
            if config['options']['finetune']:
                if dataset_str == config['options']['unlearn_dataset_name']:
                    continue
                label_name = 'label' if dataset_str != 'yahoo' else "topic"
                train_indices = get_indices(train_set, config['data'][dataset_str + '_num_class'],
                                            label_name, config['options']['finetune_nums'])
                test_indices = get_indices(test_set, config['data'][dataset_str + '_num_class'],
                                           label_name, config['options']['finetune_nums'])
                train_set = train_set.select(train_indices)
                test_set = test_set.select(test_indices)

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
    print("load {} dataset, number of batch: {}".
          format("small" if config['options']['sample_small'] else "full", len(train_dataloader)))
    return train_dataloader, eval_dataloader

