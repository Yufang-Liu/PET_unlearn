import os
import toml
from tqdm import tqdm
from argparse import ArgumentParser

import torch, evaluate

from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from model.multi_task_classfier import MultiTaskClassifier
from utils import load_config, set_seed, update
from dataset.get_dataset import get_all_dataset, get_dataset
from peft import PeftModel, PrefixTuningConfig, get_peft_model


def test(model, test_loader, device, metric):
    model.to(device)
    model.eval()
    for step, batch in enumerate(tqdm(test_loader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    eval_metric = metric.compute()
    return eval_metric['accuracy']


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
    parser.add_argument('--sample_small', action="store_true", default=False,
                        help="whether to sample small dataset or not")
    parser.add_argument("--sample_size", type=int, default=2000,
                        help="sample size for small dataset")
    parser.add_argument('--finetune', action="store_true", default=False,
                        help="whether to use finetune set, sample 10 number for each class")
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

    model = MultiTaskClassifier(config)
    for n, p in model.named_parameters():
        if "classifier_list" in n:
            p.requires_grad = False
        print(n, p.requires_grad)

    num_epochs, device = config['trainer']['max_epochs'], config['options']['device']
    metric = evaluate.load("accuracy")

    if config['options']['load_from_pretrained']:
        ckpt = torch.load(config['options']['load_from_pretrained'])
        model.load_state_dict(ckpt['model'])
        print("load from fine-tuned model, path is {}".
              format(config['options']['load_from_pretrained']))

    if config['options']['add_prefix'] and config['options']['test_only']:
        origin_acc, new_acc = {}, {}
        idx = config['data']['multi_task'].index(config['options']['unlearn_dataset_name'])
        seq_cls_model = model.get_prefix_seq_cls_model(idx)
        train_loader, test_loader = get_dataset(config['options']['unlearn_dataset_name'],
                                                config, tokenizer)

        origin_acc[config['options']['unlearn_dataset_name']] = \
            test(seq_cls_model, test_loader, device, metric)

        seq_cls_model = PeftModel.from_pretrained(
            seq_cls_model,
            config['options']['prefix_dir'])

        acc = test(seq_cls_model, test_loader, device, metric)
        new_acc[config['options']['unlearn_dataset_name']] = acc
        print("{} test finish, test acc is {}".format(config['options']['unlearn_dataset_name'],
                                                      acc))

        print("begin test other datasets")
        for test_idx, dataset_str in enumerate(config['data']['multi_task']):
            if test_idx != idx:
                seq_cls_model = model.get_prefix_seq_cls_model(test_idx)
                train_loader, test_loader = get_dataset(dataset_str,
                                                        config, tokenizer)
                origin_acc[dataset_str] = test(seq_cls_model, test_loader, device, metric)

                peft_config = None
                if config['prefix']['peft_type'] == 'prefix':
                    peft_config = PrefixTuningConfig(
                        task_type="SEQ_CLS",
                        num_virtual_tokens=config['prefix']['prefix_num'])
                seq_cls_model = get_peft_model(seq_cls_model, peft_config)
                state_dict = seq_cls_model.state_dict()
                prefix_state_dict = torch.load(config['options']['prefix_dir'] + '/adapter_model.bin')
                state_dict['prompt_encoder.embedding.weight'] = prefix_state_dict['prompt_embeddings']
                seq_cls_model.load_state_dict(state_dict)

                acc = test(seq_cls_model, test_loader, device, metric)
                new_acc[dataset_str] = acc
                print("{} test finish, test acc is {}".format(dataset_str,
                                                              acc))

        print("test finish ! original acc:")
        for k, v in origin_acc.items():
            print(k, v)
        print("after unlearn {}, new acc:".format(config['options']['unlearn_dataset_name']))
        for k, v in new_acc.items():
            print(k, v)

        exit(0)

    train_loader, test_loader = get_all_dataset(config, tokenizer)

    print(len(train_loader), len(test_loader))

    if not config['options']['add_prefix'] and config['options']['test_only']:
        model.eval()
        for step, batch in enumerate(tqdm(test_loader)):
            batch.to(device)
            with torch.no_grad():
                predictions, references = model(batch, train=False)
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        eval_metric = metric.compute()
        print("test finish, test acc is {}".format(eval_metric['accuracy']))
        exit(0)
    elif config['options']['add_prefix'] and not config['options']['test_only']:
        pass

    optimizer = AdamW(params=model.parameters(), lr=config['optim']['lr'])

    # Instantiate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_loader) * num_epochs),
        num_training_steps=(len(train_loader) * num_epochs),
    )

    model.to(device)
    best_metric, best_epoch = 0, 0
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            batch.to(device)
            loss = model(batch)
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
                predictions, references = model(batch, train=False)

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


if __name__ == '__main__':
    main()