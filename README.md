# SIGN
Code release for ICCV 2021 publication SIGN: Spatial-information Incorporated Generative Network for Generalized Zero-shot Semantic Segmentation

### 1. To use our trained models
- For Pascal VOC without self-training

`python test.py --config trained_models/voc12/configs.yaml --init_model trained_models/voc12/models_transfer/00006000.pth --val --experimentid voc12_test`

- For Pascal VOC with self-training

`python test.py --config trained_models/voc12_st/configs_st.yaml --init_model trained_models/voc12_st/models_st/00006000.pth --val --experimentid voc12_st_test --schedule st`

- For Pascal Context without self-training

`python test.py --config trained_models/context/configs.yaml --init_model trained_models/context/models_transfer/00009000.pth --val --experimentid context_test`

- For Pascal Context with self-training

`python test.py --config trained_models/context_st/configs_st.yaml --init_model trained_models/context_st/models_st/00007000.pth --val --experimentid context_st_test --schedule st`

- For COCO-stuff without self-training

`python test.py --config trained_models/coco/configs_transfer.yaml --init_model trained_models/coco/models_transfer/00009000.pth --val --experimentid coco_test --schedule mixed`

- For COCO-stuff with self-training

`python test.py --config trained_models/coco_st/configs_st.yaml --init_model trained_models/coco_st/models_st/00004000.pth --val --experimentid coco_st_test --schedule st`

### 2. To train models
Coming soon