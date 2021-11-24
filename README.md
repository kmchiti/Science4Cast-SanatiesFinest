# Science4Cast-SanatiesFinest

Scientific publications in field of AI can be viewed as an exponentially growing dynamic graph where vertices represent different concepts and each edge represents the first time two concepts got discussed and connected in a publication. Here, we demonstrate a model to predict future links in this graph based on basic and temporal properties of its vertices. We train various gradient boosting models on this data. Also, we employ dataset augmentation to make these models order-invariant. Finally, we post-process results of individual models to boost our final prediction. Our model won the 3rd place in the [Science4Cast](https://www.iarai.ac.at/science4cast/) competition.


## Requirements
Python: 3.6+

To install required packages:

```setup
pip install -r requirements.txt
```

## Reproducibility
First, download the dataset from [IARAI website](https://cloud.iarai.ac.at/index.php/s/iTx3bXgMdwsngPn) at exiting directory. Then run provide python script named boosting_train.py taking the following arguments to regenerate our top-5 submissions:


```setup
python boosting_train.py --depth=5 --estimators=400 --features=cn_pr_and_ja --lr=0.05 --minSampleSplit=14 --sampleLeaf=3 --samples=450000 --subsamples=0.5
```

```setup
python boosting_train.py --depth=5 --estimators=400 --features=pr_cn_shp_and_ccpa --lr=0.05 --minSampleSplit=18 --sampleLeaf=5 --samples=400000 --subsamples=0.7
```


```setup
python boosting_train.py --depth=4 --estimators=500 --features=pr_cn_shp_and_ja --lr=0.05 --minSampleSplit=5 --sampleLeaf=5 --samples=450000 --subsamples=0.5
```


```setup
python boosting_train.py --depth=5 --estimators=300 --features=cn_shp_and_ja_ccpa --lr=0.05 --minSampleSplit=18 --sampleLeaf=3 --samples=400000 --subsamples=0.7
```

```setup
python boosting_train.py boosting_train.py --borj2017=8 --depth=5 --estimators=500 --features=cn_shp_and_ja_ccpa --lr=0.05 --negRatio=2 --samples=400000
```
