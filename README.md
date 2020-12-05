## Convert BFM vertex colors to FLAME texture

Create a [FLAME](http://flame.is.tue.mpg.de) texture model from the Basel Face Model vertex color model.

<p align="center"> 
<img src="gifs/BFM_to_FLAME.gif">
</p>

##### About FLAME

FLAME is a lightweight and expressive generic head model learned from over 33,000 of accurately aligned 3D scans. Public FLAME related repositories:
* [TF_FLAME](https://github.com/TimoBolkart/TF_FLAME)
* [flame-fitting](https://github.com/Rubikplayer/flame-fitting)
* [Photometric FLAME Fitting](https://github.com/HavenFeng/photometric_optimization)
* [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch)
* [RingNet](https://github.com/soubhiksanyal/RingNet)
* [VOCA](https://github.com/TimoBolkart/voca)

#### Setup

Install pip and virtualenv
```
sudo apt-get install python3-pip python3-venv
```

Clone the git project:
```
$ git clone https://github.com/TimoBolkart/BFM_to_FLAME.git
```

Set up and activate virtual environment:
```
$ mkdir <your_home_dir>/.virtualenvs
$ python3 -m venv <your_home_dir>/.virtualenvs/BFM_to_FLAME
$ source <your_home_dir>/BFM_to_FLAME/bin/activate
```

Make sure your pip version is up-to-date:
```
pip install -U pip
```

Install requirements
```
pip install numpy==1.19.4
pip install h5py==3.1.0
pip install chumpy==0.70 
pip install opencv-python==4.4.0.46
```

#### Data

Download BFM 2017 (i.e. 'model2017-1_bfm_nomouth.h5') from [here](https://faces.dmi.unibas.ch/bfm/bfm2017.html) and place it in the model folder.
Download inpainting masks from [here](http://files.is.tue.mpg.de/tbolkart/FLAME/mask_inpainting.npz) and place it in the data folder.

#### Run code

Successfully running
```
python col_to_tex.py
```
outputs a 'FLAME_albedo_from_BFM.npz' in the output folder. This file can be used  with several FLAME-based repositories like [TF_FLAME](https://github.com/TimoBolkart/TF_FLAME) or [FLAME photometric optimization](https://github.com/HavenFeng/photometric_optimization).

#### Acknowledgement

We thank the authors of the BFM 2017 model for making the model publicly available. Please follow the license agreement of the BFM model when using the converted texture model.

