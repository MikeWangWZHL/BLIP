# retrieval_coco:
export CUDA_VISIBLE_DEVICES=3
python -m torch.distributed.run --nproc_per_node=1 train_retrieval_CLIP_baseline.py \
--config ./configs/retrieval_coco_eval_clip_baseline.yaml \
--output_dir output/retrieval_coco_eval_CLIP_baseline \
--evaluate