# artificial larynx vowel recognition

## set up environment
```bash
conda create -y -n ALVR python=3.9
conda activate ALVR
pip install espnet matplotlib torchvision
```


## prepare rawdata directory (csv)
```
ALVR_ROOT/
├── p1
│   ├── a/*.csv
│   ├── e/*.csv
│   ├── i/*.csv
│   ├── o/*.csv
│   └── u/*.csv
├── p2
│   ├── a/*.csv
│   ├── e/*.csv
...
...
```
Files in ALVR\_ROOT with suffix other than `csv' will be omitted

## train and test with differnent pretrained models
The following script will finetune a pretrained resnet18 model with 10 different seeds. The best and averaged accuracy on validation set will be reported:
```
bash run.sh --stage 1 --stop-stage 4 --n_seeds 10 --model resnet18 --ALVR_ROOT /path/to/ALVR_rawdata/
```
If further experiments are to be conducted with the same data, start from stage 2:
```
bash run.sh --stage 2 --stop-stage 4 --n_seeds --model densenet161
```
