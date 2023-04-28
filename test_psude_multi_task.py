import os
import toml
from tqdm import tqdm
from argparse import ArgumentParser

import torch, evaluate

from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from model.multi_task_classfier import MultiTaskClassifier
from model.modified_bert import BertForMultiTaskClassification
from transformers import BertConfig
from utils import load_config, set_seed, update
from dataset.get_dataset import get_all_dataset, get_dataset
# from peft import PeftModel, PrefixTuningConfig, get_peft_model
from local_peft.peft_model import PeftModel
from local_peft.tuners.prefix_tuning import PrefixTuningConfig
from local_peft.mapping import get_peft_model


def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file path",
                        default="./config/default.toml")
    parser.add_argument("--seed", type=int, help="random seed for experiment",
                        default=666)
    parser.add_argument('--device', type=str, default="cuda",
                        help="cpu or cuda")
    parser.add_argument('--add_prefix', action="store_true", default=False,
                        help="use pretrained prefix")
    parser.add_argument('--unlearn_dataset_name', type=str, default='dbpedia',
                        help="dataset name in [yelp, dbpedia, amazon, agnews, yahoo]")
    parser.add_argument('--prefix_dir', type=str, default=None,
                        help="pretrained prefix directory")
    parser.add_argument('--load_from_pretrained', type=str, default=None,
                        help="pretrained model file path")
    parser.add_argument('--fix_classifier', action='store_true', default=False,
                        help="whether to freeze the task classifier or not")
    parser.add_argument('--fix_prefix', action='store_true', default=False,
                        help="whether to freeze the prefix")
    parser.add_argument('--sample_small', action="store_true", default=False,
                        help="whether to sample small dataset or not")
    parser.add_argument('--finetune', action="store_true", default=False,
                        help="whether to use finetune set, sample 10 number for each class")
    parser.add_argument('--finetune_nums', type=int, default=10,
                        help="number of each class")
    parser.add_argument('--test_only', action="store_true", default=False,
                        help="whether to only test")
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
    if not os.path.exists(config['io']['ckpt_dir']):
        os.makedirs(config['io']['ckpt_dir'])
    set_seed(config['options']['seed'])

    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'],
                                              cache_dir=config['model']['cache_dir'])
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = BertForMultiTaskClassification.from_pretrained("bert-base-uncased", config)

    num_epochs, device = config['trainer']['max_epochs'], config['options']['device']
    metric = evaluate.load("accuracy")

    if config['options']['load_from_pretrained']:
        ckpt = torch.load(config['options']['load_from_pretrained'])['model']
        if list(ckpt.keys())[0].startswith('base_model'):
            print("----load model with prefix----")
            peft_config = None
            if config['prefix']['peft_type'] == 'prefix':
                peft_config = PrefixTuningConfig(
                    task_type="SEQ_MT_CLS",
                    num_virtual_tokens=config['prefix']['prefix_num'])
            else:
                print("error local_peft type !")
                exit(0)
            model = get_peft_model(model, peft_config, config)
            new_ckpt = model.state_dict()
            for k1, k2 in zip(new_ckpt.keys(), ckpt.keys()):
                assert k1 == k2
                new_ckpt[k1] = ckpt[k2]
            model.load_state_dict(new_ckpt)
        else:
            model.load_state_dict(ckpt)
        print("load from fine-tuned model, path is {}".
              format(config['options']['load_from_pretrained']))

    if config['options']['add_prefix']:
        print("---------add prefix-----------")
        peft_config = None
        if config['prefix']['peft_type'] == 'prefix':
            peft_config = PrefixTuningConfig(
                task_type="SEQ_MT_CLS",
                num_virtual_tokens=config['prefix']['prefix_num'])
        else:
            print("error local_peft type !")
            exit(0)
        model = get_peft_model(model, peft_config, config)

        if config['options']['prefix_dir']:
            adapter_model = torch.load(config['options']['prefix_dir'] + '/adapter_model.bin')
            state_dict = model.state_dict()
            state_dict['prompt_encoder.embedding.weight'] = adapter_model['prompt_embeddings']
            model.load_state_dict(state_dict)
        if config['options']['finetune']:
            for n, p in model.named_parameters():
                if "prompt_encoder" not in n and "classifier" not in n:
                    p.requires_grad = True
        if config['options']['fix_prefix']:
            for n, p in model.named_parameters():
                if "prompt_encoder" in n:
                    p.requires_grad = False

    if config['options']['fix_classifier']:
        for n, p in model.named_parameters():
            if "classifier_list" in n:
                p.requires_grad = False

    model.to(device)

    if config['options']['test_only']:
        model.eval()
        if config['options']['prefix_dir'] and config['options']['unlearn_dataset_name']:
            print("begin test other datasets after unlearn {}".
                  format(config['options']['unlearn_dataset_name']))
        else:
            print("begin test other datasets")
        for test_idx, dataset_str in enumerate(config['data']['multi_task']):
            print("test for task {} {}".format(test_idx, dataset_str))
            _, task_test_loader = get_dataset(dataset_str, config,
                                              tokenizer, add_task_id=True)
            for step, batch in enumerate(tqdm(task_test_loader)):
                batch.to(device)
                with torch.no_grad():
                    predictions, references = model(**batch)[1:]

                metric.add_batch(
                    predictions=predictions,
                    references=references,
                )

            eval_metric = metric.compute()
            print("{} test finish, test acc is {}".format(dataset_str, eval_metric['accuracy']))
        exit(0)

    if config['options']['add_prefix']:
        print("fix classifier !")
        model.print_trainable_parameters()
        print(model)
    else:
        for n, p in model.named_parameters():
            print(n, p.requires_grad)

    train_loader, test_loader = get_all_dataset(config, tokenizer)
    print(len(train_loader), len(test_loader))
    optimizer = AdamW(params=model.parameters(), lr=config['optim']['lr'])

    # Instantiate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_loader) * num_epochs),
        num_training_steps=(len(train_loader) * num_epochs),
    )

    best_metric, best_epoch = 0, 0
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            batch.to(device)
            loss = model(**batch)[0].loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'best': (best_metric, best_epoch),
            'optim': optimizer.state_dict(),
            'sched': scheduler.state_dict()
        }, config['io']['last_model'])

        model.eval()
        for step, batch in enumerate(tqdm(test_loader)):
            batch.to(device)
            with torch.no_grad():
                predictions, references = model(**batch)[1:]

            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()

        print(f"epoch {epoch}:", eval_metric)
        if eval_metric['accuracy'] > best_metric:
            best_metric, best_epoch = eval_metric['accuracy'], epoch
            os.system(f'cp {config["io"]["last_model"]} {config["io"]["best_model"]}')

    print("training finish, save models in {}, best metric is {} in epoch {}."
          .format(config['io']['best_model'], best_metric, best_epoch))


    model.eval()
    if config['options']['prefix_dir'] and config['options']['unlearn_dataset_name']:
        print("begin test other datasets after unlearn {}".
              format(config['options']['unlearn_dataset_name']))
    else:
        print("begin test other datasets")

    for test_idx, dataset_str in enumerate(config['data']['multi_task']):
        print("test for task {} {}".format(test_idx, dataset_str))
        _, task_test_loader = get_dataset(dataset_str, config,
                                          tokenizer, add_task_id=True)
        for step, batch in enumerate(tqdm(task_test_loader)):
            batch.to(device)
            with torch.no_grad():
                predictions, references = model(**batch)[1:]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        eval_metric = metric.compute()
        print("{} test finish, test acc is {}".format(dataset_str, eval_metric['accuracy']))


if __name__ == '__main__':
    main()