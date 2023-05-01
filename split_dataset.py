from argparse import ArgumentParser
from utils import load_config, update, set_seed
import toml, random
from datasets import load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict


def get_indices(dataset, num_class, label_name, sample_size):
    indices = {}
    cnt = 0
    for class_id in range(num_class):
        indices[class_id] = []

    while cnt != sample_size * num_class:
        idx = random.randint(0, len(dataset)-1)
        target_label = dataset[label_name][idx]
        if len(indices[target_label]) < sample_size:
            indices[target_label].append(idx)
            cnt += 1
            # print(cnt)
    all_indices = []
    for k, v in indices.items():
        all_indices.extend(v)
    random.shuffle(all_indices)
    return all_indices


def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="config file path",
                        default="./config/default.toml")
    parser.add_argument("--dataset_str", type=str, help="dataset name",
                        default="dbpedia")
    parser.add_argument("--seed", type=int, help="random seed for experiment",
                        default=666)
    parser.add_argument("--sample_size", type=int, default=2000,
                        help="sample size for small dataset")
    parser.add_argument("-t", "--toml", type=str, action="append")
    options = parser.parse_args()
    return options


def main():
    options = parse_argument()
    config = load_config(options.config)

    for k, v in config['io'].items():
        if v.find("{{seed}}") != -1:
            config['io'][k] = v.replace("{{seed}}", str(options.seed))

        if v.find("{{mask_word}}") != -1:
            config['io'][k] = config['io'][k]. \
                replace("{{mask_word}}", str(options.mask_word))

    if options.toml is not None:
        tomls = "\n".join(options.toml)
        new_config = toml.loads(tomls)
        update(config, new_config)

    config['options'] = {}
    for k, v in vars(options).items():
        if k not in ['config', 'toml']:
            config['options'][k] = v

    print(config)
    set_seed(config['options']['seed'])

    dataset_str = config['options']['dataset_str']
    dataset_path = config['data'][dataset_str + '_path']
    dataset_name = config['data'][dataset_str + '_dataset_name']

    train_set = load_dataset(dataset_name, cache_dir=dataset_path,
                             split='train')
    test_set = load_dataset(dataset_name, cache_dir=dataset_path,
                            split='test')
    num_class = config['data'][dataset_str + '_num_class']
    label_name = 'label' if dataset_str != 'yahoo' else "topic"
    train_indices = get_indices(train_set, num_class, label_name,
                                config['options']['sample_size'])
    test_indices = get_indices(test_set, num_class, label_name,
                               config['options']['sample_size'])
    train_set = train_set.select(train_indices)
    test_set = test_set.select(test_indices)
    dataset = DatasetDict({"train": train_set, "test": test_set})
    dataset.save_to_disk(dataset_path + '/small_'
                         + str(config['options']['sample_size'])
                         + '_' + dataset_name)
    # ds = load_from_disk(dataset_path + '/small_'
    #                     + str(config['options']['sample_size'])
    #                     + '_' + dataset_name)


if __name__ == '__main__':
    main()