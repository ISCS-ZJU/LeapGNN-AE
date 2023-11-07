import dgl
import numpy as np
import time
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from model import GATNodeFlow
from utils import mkdir_p, load_reddit

__all__ = ['run_reddit']

def setup_cuda(args):
    args['cuda'] = args['cuda'] and torch.cuda.is_available()
    if args['cuda']:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        args['device'] = 'cuda: 0'
    else:
        args['device'] = 'cpu'
    args['device'] = torch.device(args['device'])
    if torch.cuda.is_available() and not args['cuda']:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return args

def setup_reddit(args):
    new_args = {
        'batch_size': 512,
        'n_runs': 5,
        'dropout': 0.1,
        'residual': True,
        'l2_coef': 0.0,
        'patience': 10,
        'max_n_nbrs_sampled': 16,
        'gamma': 0.95,
        'lr': 0.0005,
        'n_heads': [8, 8],
        'hid_units': [64],
    }

    return new_args

def eval(device, model, nf, labels, loss_func):
    nf.copy_from_parent(ctx=device)
    soft_pred = model(nf)
    batch_nids = nf.layer_parent_nid(-1)
    batch_labels = labels[batch_nids].to(device)
    loss = loss_func(soft_pred, batch_labels)

    hard_pred = torch.max(soft_pred, dim=1)[1]
    n_correct = torch.sum(hard_pred == batch_labels)
    acc = n_correct.item() * 1.0 / len(batch_labels)

    hard_pred = hard_pred.cpu().numpy().reshape(-1,)
    batch_labels = batch_labels.cpu().numpy().reshape(-1, )
    micro_f1 = f1_score(batch_labels, hard_pred, average='micro')
    return loss, acc, micro_f1, batch_labels, hard_pred

def run_reddit(args):
    path_to_save_model = args['log_dir'] + '/trained_model'
    mkdir_p(path_to_save_model)
    checkpt_file = path_to_save_model + '/model.ckpt'

    # load and pre-process dataset
    data = load_reddit()

    # Create model
    model = GATNodeFlow(num_layers=len(args['n_heads'])-1,
                        in_dim=data['n_feats'],
                        num_hidden=args['hid_units'],
                        num_classes=data['n_classes'],
                        num_heads=args['n_heads'],
                        feat_drop=args['dropout'],
                        attn_drop=args['dropout'],
                        residual=args['residual']).to(args['device'])
    print(model)

    tb_writer = SummaryWriter(args['log_dir'])
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['l2_coef'])
    scheduler = ExponentialLR(optimizer, gamma=args['gamma'])

    dur = []
    min_loss = np.inf
    max_acc = 0.0
    max_f1 = 0.0
    n_steps = 0
    n_patient_epochs = 0
    for epoch in range(args['max_epochs']):
        if epoch >= 3:
            t0 = time.time()

        model.train()
        n_train_samples_seen = 0

        # Train
        for nf in dgl.contrib.sampling.NeighborSampler(g=data['g'],
                                                       batch_size=args['batch_size'],
                                                       expand_factor=args['max_n_nbrs_sampled'],
                                                       num_hops=len(args['n_heads']),
                                                       shuffle=True,
                                                       num_workers=16,
                                                       seed_nodes=data['train_nid'],
                                                       prefetch=True,
                                                       add_self_loop=True):
            n_steps += 1
            loss, acc, micro_f1, y_true, y_predict = \
                eval(args['device'], model, nf, data['g'].ndata['labels'], loss_fcn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            n_train_samples_seen += nf.layer_size(-1)
            print('Train samples seen {:d}/{:d} | loss {:.4f} | acc {:.4f} | micro f1 score {:.4f}\n'.format(
                n_train_samples_seen, len(data['train_nid']), loss, acc, micro_f1))

            tb_writer.add_scalar('cross entropy loss', loss, n_steps)
            tb_writer.add_scalar('accuracy', acc, n_steps)
            tb_writer.add_scalar('micro f1 score', micro_f1, n_steps)

        # Eval
        cum_batch_loss = 0
        cum_batch_n_correct = 0
        cum_y_true = []
        cum_y_pred = []
        model.eval()
        n_val_samples_seen = 0
        for nf in dgl.contrib.sampling.NeighborSampler(g=data['g'],
                                                       batch_size=args['batch_size'],
                                                       expand_factor=args['max_n_nbrs_sampled'],
                                                       num_hops=len(args['n_heads']),
                                                       shuffle=True,
                                                       num_workers=16,
                                                       seed_nodes=data['val_nid'],
                                                       prefetch=True,
                                                       add_self_loop=True):
            loss, acc, micro_f1, y_true, y_predict = \
                eval(args['device'], model, nf, data['g'].ndata['labels'], loss_fcn)

            loss = loss.item()
            n_val_samples_seen += nf.layer_size(-1)
            print('Val samples seen {:d}/{:d} | loss {:.4f} | acc {:.4f} | micro f1 score {:.4f}\n'.format(
                n_val_samples_seen, len(data['val_nid']), loss, acc, micro_f1))

            batch_size = nf.layer_size(-1)
            cum_batch_loss += loss * batch_size
            cum_batch_n_correct += acc * batch_size
            cum_y_true.append(y_true)
            cum_y_pred.append(y_predict)

        epoch_loss = cum_batch_loss / data['n_val_samples']
        epoch_acc = cum_batch_n_correct / data['n_val_samples']
        all_y_true = np.concatenate(cum_y_true, 0)
        all_y_pred = np.concatenate(cum_y_pred, 0)
        epoch_micro_f1 = f1_score(all_y_true, all_y_pred, average='micro')
        tb_writer.add_scalar('epoch val cross entropy loss', epoch_loss, epoch)
        tb_writer.add_scalar('epoch val accuracy', epoch_acc, epoch)
        tb_writer.add_scalar('epoch val micro f1', epoch_micro_f1, epoch)

        if epoch_micro_f1 > max_f1:
            max_f1 = epoch_micro_f1
            torch.save(model.state_dict(), checkpt_file)
            n_patient_epochs = 0
        else:
            n_patient_epochs += 1
        print('Number of epochs with no improvement on training set '
              'measured by f1 score: {:d}'.format(n_patient_epochs))

        if epoch_acc > max_acc:
            max_acc = epoch_acc

        if epoch_loss < min_loss:
            min_loss = epoch_loss

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f}".format(epoch + 1, np.mean(dur)))
        print("=================================================================")
        print("current loss {:.4f} | min loss {:.4f}".format(epoch_loss, min_loss))
        print("current accuracy {:.4f} | max accuracy {:.4f}".format(epoch_acc, max_acc))
        print("current f1 score {:.4f} | max f1 score {:.4f}".format(epoch_micro_f1, max_f1))
        print("=================================================================\n")

        if n_patient_epochs == args['patience']:
            print('Early stop after {:d} epochs!'.format(epoch + 1))
            break
        scheduler.step()

    model.load_state_dict(torch.load(checkpt_file))
    model.eval()

    cum_test_loss = 0
    cum_test_n_correct = 0
    cum_test_y_true = []
    cum_test_y_pred = []

    for nf in dgl.contrib.sampling.NeighborSampler(g=data['g'],
                                                   batch_size=args['batch_size'],
                                                   expand_factor=args['max_n_nbrs_sampled'],
                                                   num_hops=len(args['n_heads']),
                                                   shuffle=True,
                                                   num_workers=16,
                                                   seed_nodes=data['test_nid'],
                                                   prefetch=True,
                                                   add_self_loop=True):
        test_loss, test_acc, test_f1, test_true, test_predict = \
            eval(args['device'], model, nf, data['g'].ndata['labels'], loss_fcn)
        batch_size = nf.layer_size(-1)
        cum_test_loss += test_loss.item() * batch_size
        cum_test_n_correct += test_acc * batch_size
        cum_test_y_true.append(test_true)
        cum_test_y_pred.append(test_predict)

    test_loss = cum_test_loss / data['n_test_samples']
    test_acc = cum_test_n_correct / data['n_test_samples']
    all_test_y_true = np.concatenate(cum_test_y_true, 0)
    all_test_y_pred = np.concatenate(cum_test_y_pred, 0)
    epoch_micro_f1 = f1_score(all_test_y_true, all_test_y_pred, average='micro')

    print("Test results")
    print("=================================================================")
    print("loss {:.4f} | acc {:.4f} | micro f1 {:.4f}".format(test_loss, test_acc, epoch_micro_f1))
    print("=================================================================")

    return {'test_loss': test_loss,
            'test_acc': test_acc,
            'test_micro_f1': epoch_micro_f1,
            'total time': np.sum(dur),
            'mean_epoch_time': np.mean(dur),
            'num_epochs': epoch + 1}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Attention Understanding')
    args = parser.parse_args().__dict__

    default_args = {
        'cuda': True,
        'dataset': 'reddit',
        'patience': 30,
        'max_epochs': 200
    }
    args.update(default_args)
    args = setup_cuda(args)
    args.update(setup_reddit(args))
    args['log_dir'] = 'results/{}'.format(args['dataset'])
    mkdir_p(args['log_dir'])
    run_reddit(args)
