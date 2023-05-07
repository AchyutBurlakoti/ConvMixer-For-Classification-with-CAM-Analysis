# ConvMixer-For-Classification-with-CAM-Analysis
First download the dataset from : [https://drive.google.com/file/d/174Ffina6HoFF731a8DgMJs0Wm5OHQ0-S/view?usp=share_link](https://drive.google.com/file/d/174Ffina6HoFF731a8DgMJs0Wm5OHQ0-S/view?usp=sharing) then unzip the dataset in the root dir.

Download the pre-trained models from : https://drive.google.com/drive/folders/1QCsXRoqMQQvof1ZzuB0OYyMGvaJ7SghL?usp=sharing then put those models in the models directory inside the root directory.

Please note that label for an image is extracted from the folder's name by parsing the file path as written in dataset.py file. The parsing is done for windows operating system so splitting code might be different for linux, Google Colab or other environments.

To train ConvMixer :- 
* set MODEL_NAME = 'conv-mix' inside train.py then
* Run following command at root dir.
```
python train.py
```

To train ResNet-50 :- 
* set MODEL_NAME = 'res-net' inside train.py then
* Run following command at root dir.
```
python train.py
```

The notebooks contains the testing of the trained models along with the comparsions outputs.

### A typical top-level directory layout

    ├── archive                 
        ├── test
            ├── Parasitized
            ├── Uninfected
        ├── train
            ├── Parasitized
            ├── Uninfected
    ├── models
        ├── conv-mix.pt
        ├── res-net.pt
    ├── train.py    # to train the model                   
    ├── datset.py   # custom dataset class
    ├── utils.py
    ├── conv-mix-gap-layer-analysis.ipynb      # notebook which shows cam analysis of convmix model
    ├── convmix-precision-recall-test.ipynb   # notebook showing precision and recall score
    ├── resnet-gap-layer-analysis.ipynb       # cam analysis of resnet-50 model
    ├── resnet-precision-recall-test.ipynb    # precision and recall analysis of resnet-50 model
    ├── LICENSE
    └── README.md
