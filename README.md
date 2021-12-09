# AUTOMAP

Welcome to the official repository for AUTOMAP (Automated Transform by Manifold Approximation), a Tensorflow (> 2.0) implementation of the model described in:

[Article PDF](http://martinos.org/lfi/pdf/AUTOMAP_Nature_2018.pdf): B. Zhu, J. Z. Liu, S. F. Cauley, B. R. Rosen, and M. S. Rosen, “Image reconstruction by domain-transform manifold learning,” Nature, vol. 555, no. 7697, pp. 487 EP ––492, Mar. 2018


[MRI Training and Testing Data](https://www.dropbox.com/sh/fy5gnn6t1c6qgl2/AAAqIBMIaAlr4ZKLby-9u4QSa?dl=1) The data_64 folder contains, 64x64 images, 'train_input.mat' for training AUTOMAP with labels 'train_x_real.mat' and 'train_x_img.mat' for training each neural network respectively. There are also test images with similar input and output name formats. Please download these files and change the data directory inputs in the configs folder to your local directories. The data files were generated using Matlab code and thus are .mat format but this is not a requirement. You can easily adjust the automap_data_generator.py and automap_inference_data_generator.py to load numpy data (or other formats).

This AUTOMAP version was updated (11/9/2021) to match the legacy hardcoded loss function.

## Training

Training utilizes `automap_main_train.py` and a JSON config file.

```
python automap_main_train.py -c configs/train_64x64_ex.json
```

Some key config file entries:

"exp_name": Name of experiment. Intermediate model files will be stored in `experiments/{exp_name}`

"resume": Resume training (based on "model_load" file)". 1 or 0

"loadmodel_dir": File location of model to be loaded (for resume = 1). If not resuming a previous training run, it should be set to null.

"num_epochs": Number of training epochs

"learning_rate": Learning rate

"batch_size": Size of training mini-batch.

"fc_input_dim": Input dimensionality (# nodes)

"fc_hidden_dim": Fully-Connected Hidden layer dimensionality ((# nodes)

"fc_output_dim": Fully-Connected Output layer dimensionality (# nodes)

"im_h": 128: Output image size (height in pixels)

"im_w": 128: Input image size (height in pixels)

"data_dir": Directory of Training Data where "train_input" and "train_output" are located.

"train_input": Input data file (as .mat file) , "train_input.mat"

"train_output": Output data file (as .mat file) , "train_x_real.mat" or "train_x_img.mat"


## Inference

Inference utilizes `automap_main_inference.py` and a JSON config file.

```
python automap_main_inference.py -c configs/inference_64x64_ex.json
```

The JSON inference config file uses similar entries as the training config file mentioned above, but instead of "train_input" and "train_output" it takes in "inference_input"
