# ADL Final Project
Generative Embedding Inversion Attack (GEIA)

## run baseline GEIA
```
python baseline_geia.py --dataset=qnli --train_ratio=0.05
```

## produce easy data augmentation data
multiple is for the augmentation size
count is for the number of augmenting words
```
python eda_produce_data.py --dataset=qnli --train_ratio=0.05 --multiple=1 --count=3
```

## run GEIA on easy data augmentation data
option is for the appoarch of easy data augmentation
```
python eda_geia.py --dataset=qnli --train_ratio=0.05 --option=swap --multiple=1 --count=1
```

## plot the experienment result
run plot.ipynb

## others
folder research is for some tools, including config, model, data, utils