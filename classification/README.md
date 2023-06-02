## Dataset Preparation

Download the ImageNet 2012 dataset from [here](http://image-net.org/), and prepare the dataset based on this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4). The file structure should look like:

```
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```



## Training


To train a model on ImageNet:

```bash
python -m torch.distributed.launch --nproc_per_node [num_gpus] --master_port 13335  main.py --cfg [path/to/config]
```

For example, train FAT-b2 with 8 GPUs by

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 13335  main.py --cfg configs/FAT_b2.yaml
```

## Results

|Model | Resolution | Params(M) |FLOPs(G) | Top-1(%)|
|------|-----------|-----------|---------|--------|
|FAT-B0|224        | 4.5       |0.7      | 77.6   |
|FAT-B1|224        | 7.8       |1.2      | 80.1   |
|FAT-B2|224        | 13.5      |2.0      |81.9    |
|FAT-B3|224        | 29        |4.4      |83.6    |



