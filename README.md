## Learning Image Restoration

Noisy image | Restored image
:--:|:--:
![Noisy image](examples/noisy.png) | ![Restored image](examples/restored.png)

Pytorch unofficial implementation of *[On learning optimized reaction diffusion processes for effective image restoration](https://arxiv.org/abs/1503.05768)* by Yunjin Chen, Wei Yu and Thomas Pock.

---

You can train your own model :
```bash
python train.py models/new_model.pt
```

Or test an already trained one :
```bash
python test.py models/joint7.pt
```

### Experiments
As it was done before and in this paper, I used the [BSDS300](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/) image dataset.
Then I implemented Salt and pepper noise (SnP), or used additive noise from skimage library (s&p, Poisson and speckle noise) in addition to the gaussian noise that was already done

The different models trained considered were : 
- TRD with 8 stages, 24 filters of size 5x5 (stored in `models/noise_type_joint5.pt`)
- TRD with 8 stages, 48 filters of size 7x7 (stored in `models/noise_type_joint7.pt`)

An Adam optimizer was used instead of the L-BFGS algorithm. During training, 180x180 patches randomly cropped from training images are fed to the network (GPU memory limitations might force you to reduce the size).

The learning phase uses backpropagation from Pytorch on the GPU whereas authors' MATLAB implementation used explicit derivatives on a CPU.

TRD models were first trained greedily as explained in the paper (see `models/greedyx.pt`). Then, they were finetuned using the joint method (see `models/jointx.pt`).

Here are the results of the experiment on the test set :

For Gaussian Noise

| PSNR (dB) |                  |                  |                  |                  |
|-----------|:----------------:|:----------------:|:----------------:|:----------------:|
|           |       σ=15       |                  |       σ=25       |                  |
| **Stage**     | `models/*joint5.pth` | `models/*joint7.pth` | `models/*joint5.pth` | `models/*joint7.pth` |
| 2         |       31.23      |       27.49      |       27.71      |       24.92      |
| 5         |       31.31      |       29.90      |       29.15      |       27.65      |
| 8         |       30.94      |       31.23      |       **30.11**      |       **30.26**      |

The results are better for σ=25 than the paper. Models were  trained only for 2-3 hours whereas the authors spent almost 24h on one model. Results can be improved easily with more time.


For our implementation of Salt and Pepper noise

| PSNR (dB) |                  |                  |                  |                  |
|-----------|:----------------:|:----------------:|:----------------:|:----------------:|
|           |       threshold=0.0005       |                  |       threshold=0.001       |                  |       threshold=0.005       |                  |
| **Stage**     | `models/*joint5.pth` | `models/*joint7.pth` | `models/*joint5.pth` | `models/*joint7.pth` | `models/*joint5.pth` | `models/*joint7.pth` |
| 2         |       18.53      |       18.58      |       18.52      |       18.56      |       18.51      |       18.54      |
| 5         |       19.26      |       20.08      |       19.27      |       20.06      |       19.25      |       20.07      |
| 8         |       20.69      |       20.93      |       20.68      |       20.91      |       20.66      |       20.89      |


For S&P noise by skimage

| PSNR (dB) |                  |                  |                  |                  |
|-----------|:----------------:|:----------------:|:----------------:|:----------------:|
|           |       amount=0.05       |                  |       amount=0.1       |                  |       amount=0.2       |                  |
| **Stage**     | `models/*joint5.pth` | `models/*joint7.pth` | `models/*joint5.pth` | `models/*joint7.pth` | `models/*joint5.pth` | `models/*joint7.pth` |
| 2         |       24.54      |       23.92      |       24.56      |       24.66      |       23.21      |       24.03      |
| 5         |       24.50      |       23.88      |       25.03      |       24.78      |       24.50      |       24.87      |
| 8         |       24.63      |       23.95      |       25.28      |       24.84      |       25.00      |       25.08      |


For Speckle noise

| PSNR (dB) |                  |                  |                  |                  |
|-----------|:----------------:|:----------------:|:----------------:|:----------------:|
|           |       σ=15       |                  |       σ=25       |                  |
| **Stage**     | `models/*joint5.pth` | `models/*joint7.pth` | `models/*joint5.pth` | `models/*joint7.pth` |
| 2         |       23.67      |       25.79      |       22.62      |       24.63      |
| 5         |       25.49      |       27.25      |       24.72      |       26.41      |
| 8         |       28.65      |       28.63      |       28.16      |       28.15      |


For Poisson noise

| PSNR (dB) |                  |                  |
|-----------|:----------------:|:----------------:|
| **Stage**     | `models/*joint5.pth` | `models/*joint7.pth` |
| 2         |       12.60      |       10.46      |
| 5         |       14.39      |       20.51      |
| 8         |       34.26      |       34.47      |
