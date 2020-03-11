# orderless-rnn-classification
This repository provides the MS-COCO training code for the [Orderless Recurrent Models for Multi-label Classification](https://arxiv.org/abs/1911.09996) paper which will be published in CVPR2020.

## Environment
* python 2.7
* pytorch 0.4.1

## Training
Three steps of training:
### Encoder
`python train_bce.py -lr 1e-3 -momentum 0.9 -image_path {image_path} -save_path {save_path1}`
### Encoder + decoder
`python train_lstm.py -image_path {image_path} -save_path {save_path2} -order_free pla -finetune_encoder -swa_params "{'lr_high': 1e-3, 'lr_low': 1e-6, 'cycle_length': 3, 'swa_coeff': 0.1}" -encoder_weights {save_path1}/BEST_checkpoint.pt`
### Decoder
`python train_lstm.py -image_path {image_path} -save_path {save_path3} -order_free pla -decoder_lr 1e-5 -snapshot {save_path2}/BEST_checkpoint.pth.tar -epochs 5 -train_from_scratch`

## Testing
`python train_lstm.py -image_path {image_path} -snapshot {save_path3}/BEST_checkpoint.pth.tar -test_model`

## Acknowledgements
The encoder-decoder architecture that is implemented in this repository is based on the implementation in [here](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/).
