python test.py --config trained_models/context_st/configs_st.yaml --init_model trained_models/context_st/models_st/00007000.pth --val --experimentid context_st_test --schedule st

python test.py --config trained_models/coco/configs_transfer.yaml --init_model trained_models/coco/models_transfer/00009000.pth --val --experimentid coco_test --schedule st

python test.py --config trained_models/coco_st/configs_st.yaml --init_model trained_models/coco_st/models_st/00004000.pth --val --experimentid coco_st_test --schedule st