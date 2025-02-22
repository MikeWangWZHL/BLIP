# retrieval_coco:
export CUDA_VISIBLE_DEVICES=0,3
python -m torch.distributed.run --nproc_per_node=2 train_retrieval.py \
--config ./configs/retrieval_coco_mine.yaml \
--output_dir output/retrieval_coco \
--evaluate