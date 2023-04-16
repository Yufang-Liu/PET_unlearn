import os
import toml
from tqdm import tqdm
from argparse import ArgumentParser

import torch, evaluate

from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from model.multi_task_classfier import MultiTaskClassifier
from utils import load_config, set_seed, update
from dataset.get_dataset import get_all_dataset


def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file path",
                        default="./config/default.toml")
    parser.add_argument("--seed", type=int, help="random seed for experiment",
                        default=666)
    parser.add_argument('--device', type=str, default="cuda",
                        help="cpu or cuda")
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

    train_loader, test_loader = get_all_dataset(config, tokenizer)

    print(len(train_loader), len(test_loader))

    model = MultiTaskClassifier(config)
    # multi_task_train(config, model, train_loader, test_loader, num_epochs=5)

    num_epochs, device = config['trainer']['max_epochs'], config['options']['device']
    metric = evaluate.load("accuracy")

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