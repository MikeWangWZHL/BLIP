# retrieval_coco:
export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.run --nproc_per_node=1 --master_port 29501 train_retrieval.py \
--config ./configs/retrieval_coco_finetune_custom_subset_train.yaml \
--output_dir output/retrieval_coco_finetune_custom_subset_train