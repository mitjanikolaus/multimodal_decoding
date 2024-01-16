# multimodal_decoding

## VisualBERT feature extraction

```
pip install opencv-python pyyaml==5.1
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## LXMERT features

Precomputed bottom-up features can be downloaded from [this link](https://storage.googleapis.com/up-down-attention/trainval.zip)
(Source: https://github.com/peteanderson80/bottom-up-attention).

The downloaded features need to be extracted to `~/data/coco/bottom_up_feats/`.