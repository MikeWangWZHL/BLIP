# retrieval_coco:
# export CUDA_VISIBLE_DEVICES=0,3
# python -m torch.distributed.run --nproc_per_node=2 train_retrieval.py \
# --config ./configs/retrieval_coco_custom.yaml \
# --output_dir output/retrieval_coco_subset_custom \
# --evaluate

export CUDA_VISIBLE_DEVICES=1,2
export WORLD_SIZE=2

python -m torch.distributed.run --nproc_per_node=2 train_retrieval.py \
--config ./configs/retrieval_coco_custom.yaml \
--output_dir output/retrieval_coco_subset_custom \
--device cuda:1