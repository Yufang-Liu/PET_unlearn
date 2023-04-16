import os
import toml
from tqdm import tqdm
from argparse import ArgumentParser

from utils import load_config, set_seed, update
from dataset.get_dataset import get_dataset
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import evaluate
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)


def kl_loss(teacher_output, student_output, temperature=1):
    logprobs = F.log_softmax(student_output, dim=-1)
    soft_targets = torch.softmax(teacher_output / temperature, dim=-1)
    loss = F.kl_div(logprobs / temperature, soft_targets, reduction='sum')
    return loss


def hard_cross_entropy_loss(teacher_output, student_output):
    teacher_predict = torch.argmax(teacher_output, dim=1)
    loss = F.cross_entropy(student_output, teacher_predict)
    return loss


def run_teacher_student_learning(config, device, model, teacher_model,
                                 train_loader, test_loader, metric,
                                 optimizer, scheduler, num_epochs):
    set_seed(config['options']['seed'])
    best_acc, best_epoch = 0, 0
    model.to(device)
    teacher_model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            batch.to(device)

            with torch.no_grad():
                teacher_output = teacher_model(**batch).logits.detach()

            optimizer.zero_grad()
            outputs = model(**batch).logits
            if config['prefix']['distill_loss'] == 'hard':
                loss = hard_cross_entropy_loss(teacher_output, outputs)
            else:
                loss = kl_loss(teacher_output, outputs)

            loss.backward()
            optimizer.step()
            scheduler.step()

            predictions = outputs.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        train_acc = metric.compute()['accuracy']

        model.save_pretrained(config['io']['last_prefix_model'])

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

        acc = metric.compute()['accuracy']

        print("epoch {}: train acc: {} test acc: {}".format(epoch, train_acc, acc))

        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            os.system(f'cp -r {config["io"]["last_prefix_model"]} {config["io"]["best_prefix_model"]}')


def run_normal_training(config, device, model,
                        train_loader, test_loader, metric,
                        optimizer, scheduler, num_epochs):
    model.to(device)
    best_metric, best_epoch = 0, 0
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_loader)):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
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
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
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


def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file path",
                        default="./config/default.toml")
    parser.add_argument("--seed", type=int, help="random seed for experiment",
                        default=666)
    parser.add_argument('--add_prefix', action="store_true", default=False,
                        help="use prefix for learning")
    parser.add_argument('--fix_classifier', action='store_true', default=False,
                        help="whether to freeze the task classifier or not")
    parser.add_argument('--load_from_pretrained', type=str, default=None,
                        help="pretrained model file path")
    parser.add_argument('--device', type=str, default="cuda",
                        help="cpu or cuda")
    parser.add_argument('--dataset_name', type=str, default='dbpedia',
                        help="dataset name in [yelp, dbpedia, amazon, agnews, yahoo]")
    parser.add_argument('--sample_small', action="store_true", default=False,
                        help="whether to sample small dataset or not")
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

    dataset_str = config['options']['dataset_name']
    model = AutoModelForSequenceClassification.\
        from_pretrained(config['model']['model_name'],
                        return_dict=True,
                        num_labels=config['data'][dataset_str+'_num_class'],
                        cache_dir=config['model']['cache_dir'])
    if config['options']['load_from_pretrained']:
        ckpt = torch.load(config['options']['load_from_pretrained'])
        model.load_state_dict(ckpt['model'])
        print("load from fine-tuned model, path is {}".
              format(config['options']['load_from_pretrained']))

    if config['options']['add_prefix']:
        print("---------add prefix-----------")
        peft_config = None
        if config['prefix']['peft_type'] == 'prefix':
            peft_config = PrefixTuningConfig(
                task_type="SEQ_CLS",
                num_virtual_tokens=config['prefix']['prefix_num'])
        else:
            print("error peft type !")
            exit(0)
        model = get_peft_model(model, peft_config)
        if config['options']['fix_classifier']:
            for n, p in model.named_parameters():
                if 'classifier' in n:
                    p.requires_grad = False
        model.print_trainable_parameters()
        print(model)

    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'],
                                              cache_dir=config['model']['cache_dir'])
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_loader, test_loader = get_dataset(dataset_str, config, tokenizer)
    num_epochs, device = config['trainer']['max_epochs'], config['options']['device']
    metric = evaluate.load("accuracy")

    optimizer = AdamW(params=model.parameters(), lr=config['optim']['lr'])

    # Instantiate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_loader) * num_epochs),
        num_training_steps=(len(train_loader) * num_epochs),
    )

    if not config['options']['add_prefix']:
        run_normal_training(config, device, model, train_loader, test_loader, metric,
                            optimizer, scheduler, num_epochs)
    else:
        teacher_model = AutoModelForSequenceClassification.\
            from_pretrained(config['model']['model_name'],
                            return_dict=True,
                            num_labels=config['data'][dataset_str + '_num_class'],
                            cache_dir=config['model']['cache_dir'])
        run_teacher_student_learning(config, device, model, teacher_model,
                                     train_loader, test_loader, metric,
                                     optimizer, scheduler, num_epochs)




if __name__ == '__main__':
    main()






