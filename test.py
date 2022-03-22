import torch
import argparse
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from data import BerkeleyLoader
from model import DiffusionNet


def parse_args():
    parser = argparse.ArgumentParser(description='Test a model.')
    parser.add_argument('--noise_type', '-nt',  choices=['Gaussian', 'SnP', 's&p', 'speckle', 'poisson'], default='Gaussian', help='Type of additive noise for training')
    parser.add_argument('--steps', '-T', type=int, default=8, help='Number of steps of the net')
    parser.add_argument('--n_filters', '-nf', type=int, default=48, help='Number of filters')
    parser.add_argument('--filter_size', '-fs', type=int, default=7, help='Size of the filters')
    parser.add_argument('model_path', help='Path to the model')
    args = parser.parse_args()
    return args


def calc_psnr(im_true, im):
    im_true = im_true.detach().cpu().numpy()[0].transpose((1,2,0))
    im = im.detach().cpu().numpy()[0].transpose((1,2,0))
    m = psnr(im_true, im)
    return m

def evaluation(model,noise_type, parameter):
    print("parameter :", parameter)
    test_loader = BerkeleyLoader(noise_type=noise_type, parameter=parameter, train=False, num_workers=2)
    psnr_steps = [0] * len(model.dnets)
    for im_noisy, im in tqdm(test_loader):
        im_noisy, im = im_noisy.cuda(), im.cuda()
        im_pred = torch.clone(im_noisy)
        for i in range(len(model.dnets)):
            with torch.no_grad():
                im_pred = model.step(im_pred, im_noisy, i)
            psnr_steps[i] += calc_psnr(im, im_pred)
    print("Avrg PSNR")
    for i in range(len(psnr_steps)):
        print(i+1, "{:.2f} dB".format(psnr_steps[i]/len(test_loader)))



def test(model, noise_type):
    model.eval()
    parameter={}
    if noise_type =='Gaussian' or noise_type=='speckle':
      for sigma in [15, 25]:
          parameter['sigma']=sigma
          evaluation(model, noise_type, parameter)


    if noise_type =='SnP':
      for threshold in [0.0005, 0.001, 0.005]:
          parameter['threshold']=threshold
          evaluation(model, noise_type, parameter)

    if noise_type =='poisson':
        evaluation(model, noise_type, parameter)
    if noise_type =='s&p':
      for amount in [0.05, 0.1, 0.2]:
          parameter['amount']=amount
          evaluation(model, noise_type, parameter)


if __name__ == '__main__':
    args = parse_args()
    model = DiffusionNet(T=args.steps, n_rbf=63, n_channels=3, n_filters=args.n_filters, filter_size=args.filter_size).cuda()
    model.load_state_dict(torch.load(args.model_path))
    test(model, args.noise_type)