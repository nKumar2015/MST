accelerate launch --config_file audio-diffusion/config/accelerate_local.yaml \
audio-diffusion/scripts/train_unet.py \
--dataset_name Nkumar5/MST \
--output_dir output \
--num_epochs 100 \
--train_batch_size 8 \
--eval_batch_size 8 \
--gradient_accumulation_steps 8 \
--learning_rate 1e-4 \
--lr_warmup_steps 500 \
--mixed_precision no \
--push_to_hub True \
--hub_model_id NKumar5/Conditional_Audio_Diffusion \
--encodings data/encodings.p \
