# retrieval_coco:
export CUDA_VISIBLE_DEVICES=0,3
python -m torch.distributed.run --nproc_per_node=2 train_retrieval_with_grouping.py \
--config configs/retrieval_coco_finetune_with_grouping_custom_subset_training_srl.yaml \
--output_dir output/retrieval_coco_finetune_with_grouping_custom_subset_training_srl