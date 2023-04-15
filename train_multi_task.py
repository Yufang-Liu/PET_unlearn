import os
import pickle
import toml
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from model.multi_task_classfier import MultiTaskClassifier
from utils import load_config, set_seed, update


def parse_argument():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="config file path",
                        default="./config/default.toml")
    parser.add_argument("--seed", type=int, help="random seed for experiment",
                        default=666)
    parser.add_argument('--device', type=str, default="cuda",
                        help="cpu or cuda")
    parser.add_argument('--sample_small', action="store_true", default=False,
                        help="whether to sample small dataset or not")
    parser.add_argument("-t", "--toml", type=str, action="append")
    options = parser.parse_args()
    return options


def multi_task_test(name, config, model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    loss_fn = F.cross_entropy
    set_seed(config['options']['seed'])
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            for x in batch:
                if x != 'task':
                    batch[x] = batch[x].cuda()
            targets = batch['labels']
            outputs = model(batch)
            loss = loss_fn(outputs, targets, reduction='sum')

            test_loss += loss.item()
            predicted = torch.argmax(outputs, dim=-1)
            total += targets.size()[0]
            correct += predicted.eq(targets.long()).sum().item()
    acc = 100.0 * correct / total
    print('\n{}: Average loss: {:.3f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        name, test_loss / total, correct, total, acc))

    return acc


def multi_task_train(config, model, data_loader, test_loader, num_epochs):
    best_acc, best_epoch = 0.0, 0
    set_seed(config['options']['seed'])
    loss_fn = F.cross_entropy
    num_step = len(data_loader) // 10
    model.set_optimizer()
    for epoch in range(num_epochs):
        model.train()
        correct, total, train_loss = 0, 0, 0
        for batch_idx, batch in enumerate(data_loader):
            for x in batch:
                if x != 'task':
                    batch[x] = batch[x].cuda()
            model.optim.zero_grad()
            outputs = model(batch)
            loss = loss_fn(outputs, batch['labels'])

            loss.backward()
            model.optim.step()

            train_loss += loss.item()
            targets = batch['labels']
            predicted = torch.argmax(outputs, dim=-1)
            total += targets.size()[0]
            correct += predicted.eq(targets.long()).sum().item()
            acc = 100. * correct / total

            if hasattr(model, 'sched'):
                model.sched.step()

            if batch_idx % num_step == 0:
                print('Train Epoch: {} [{}/{} ({:.3f}%)]\tLoss: {:.3f}\tAcc: {:.3f}%'.format(
                    epoch, batch_idx * len(batch['labels']), len(data_loader.dataset),
                           100. * batch_idx / len(data_loader), loss.item(), acc))

        print("learning rate in epoch {}: lr:{}".
              format(epoch + 1, model.optim.state_dict()['param_groups'][0]['lr']))

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'best': (best_acc, best_epoch),
            'optim': model.optim.state_dict(),
            'sched': model.sched.state_dict()
            if hasattr(model, 'sched') else {},
        }, config['io']['last_model'])

        acc = multi_task_test("finetune on train_loader", config, model, test_loader)

        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            os.system(f'cp {config["io"]["last_model"]} {config["io"]["best_model"]}')


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
    multi_task_train(config, model, train_loader, test_loader, num_epochs=5)


if __name__ == '__main__':
    main()