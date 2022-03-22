import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd.profiler as profiler
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from data import BerkeleyLoader
from model import DiffusionNet


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('--noise_type', '-nt',  choices=['Gaussian', 'SnP', 's&p', 'speckle', 'poisson'], default='Gaussian', help='Type of additive noise for training')
    parser.add_argument('--sigma', type=int, default=25, help='Standard deviation of the additive gaussian noise used for training')
    parser.add_argument('--amount', type=float, default=0.2, help='Value for amount of salt and pepper pixel using skimage')
    parser.add_argument('--salt_vs_pepper','-sp', type=float, default=0.5, help='Value for proportion of salt/pepper')
    parser.add_argument('--threshold', type=int, default=0.005, help='threshold for SnP noise')
    parser.add_argument('--lowervalue', '-lv', type=int, default=5, help='Lower value for SnP noise')
    parser.add_argument('--uppervalue', '-uv', type=int, default=250, help='Upper value for SnP noise')

    parser.add_argument('--steps', '-T', type=int, default=8, help='Number of steps of the net')
    parser.add_argument('--n_filters', '-nf', type=int, default=48, help='Number of filters')
    parser.add_argument('--filter_size', '-fs', type=int, default=7, help='Size of the filters')
    parser.add_argument('--mode', choices=['greedy', 'joint'], default='greedy', help='Training mode')
    parser.add_argument('--pretrained', default=None, help='Path of the pretrained model. If None, create one')
    parser.add_argument('--epochs', '-e', type=int, default=6, help='Number of epochs')
    parser.add_argument('model_path', help='Path to the model')
    args = parser.parse_args()
    return args


def calc_psnr(im_true, im_noisy):
    im_true = im_true.detach().cpu().numpy()
    im_noisy = im_noisy.detach().cpu().numpy()
    return psnr(im_true, im_noisy)

def training(model, criterion, optimizer, scheduler, epochs, train_loader, test_loader, mode,path):
    best_psnr = 0
    not_improvement_counter = 0
    for e in range(epochs):
        print("Epoch", e)

        # Training the model
        print("Training")
        model.train()
        loss_train = 0
        for im_noisy, im in tqdm(train_loader):
            im_noisy, im = im_noisy.cuda(), im.cuda()
            im_pred = torch.clone(im_noisy)
            if mode == "greedy":
                for i in range(len(model.dnets)):
                    optimizer.zero_grad()
                    im_pred = model.step(im_pred, im_noisy, i)
                    loss = criterion(im_pred, im)
                    loss.backward()
                    optimizer.step()
                    im_pred = im_pred.detach()
            elif mode == "joint":
                im_pred = model(im_pred, im_noisy)
                optimizer.zero_grad()
                loss = criterion(im_pred, im)
                loss.backward()
                optimizer.step()
            loss_train += loss.item()
        print("Training loss : ", loss_train/len(train_loader))
        
        # Evaluating the model
        if e % 5 == 0 or e == epochs-1:
            print("Testing")
            model.eval()
            loss_val = 0
            metric = 0
            for im_noisy, im in tqdm(test_loader):
                im_noisy, im = im_noisy.cuda(), im.cuda()
                im_pred = torch.clone(im_noisy)
                with torch.no_grad():
                    im_pred = model(im_pred, im_noisy)
                    loss = criterion(im_pred, im)
                metric += calc_psnr(im, im_pred)
                loss_val += loss.item()
            avrg_val_loss = loss_val/len(test_loader)
            avrg_metric = metric / len(test_loader)
            print("Validation loss : ", avrg_val_loss)
            print("PSNR : ", avrg_metric)
        
            if avrg_metric > best_psnr:
                print("Best PSNR. Saving model")
                best_psnr = avrg_metric
                torch.save(model.state_dict(), path)
            scheduler.step(avrg_metric)


        ### early stop

        if e == 0:
          best_loss= loss_train/len(train_loader)
        else :
          if best_loss > loss_train/len(train_loader):
            best_loss= loss_train/len(train_loader)
            not_improvement_counter = 0
          else :
            not_improvement_counter +=1
        if not_improvement_counter >=6:
          print("Testing")
          model.eval()
          loss_val = 0
          metric = 0
          for im_noisy, im in tqdm(test_loader):
              im_noisy, im = im_noisy.cuda(), im.cuda()
              im_pred = torch.clone(im_noisy)
              with torch.no_grad():
                  im_pred = model(im_pred, im_noisy)
                  loss = criterion(im_pred, im)
              metric += calc_psnr(im, im_pred)
              loss_val += loss.item()
          avrg_val_loss = loss_val/len(test_loader)
          avrg_metric = metric / len(test_loader)
          print("Validation loss : ", avrg_val_loss)
          print("PSNR : ", avrg_metric)
      
          if avrg_metric > best_psnr:
              print("Best PSNR. Saving model")
              best_psnr = avrg_metric
              torch.save(model.state_dict(), path)
          scheduler.step(avrg_metric)
          break
    
    return None

if __name__ == '__main__':
    args = parse_args()
    parameter={'sigma':args.sigma,
                'threshold':args.threshold,'lowerValue':args.lowervalue, 'upperValue':args.uppervalue,
                'amount':args.amount,'salt_vs_pepper':args.salt_vs_pepper}
    train_loader = BerkeleyLoader(noise_type=args.noise_type, parameter=parameter, train=True, batch_size=1, shuffle=True, num_workers=2)
    test_loader = BerkeleyLoader(noise_type=args.noise_type, parameter=parameter, train=False, num_workers=2)
    model = DiffusionNet(T=args.steps, n_rbf=63, n_channels=3, n_filters=args.n_filters, filter_size=args.filter_size).cuda()
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=0, factor=0.1, verbose=True)
    training(model, criterion, optimizer, scheduler, args.epochs, train_loader, test_loader, args.mode, args.model_path)
