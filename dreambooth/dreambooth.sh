#! /bin/bash 

nvidia-smi
nvcc --version

hostname
NUM_GPU_AVAILABLE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo "number of gpus:" $NUM_GPU_AVAILABLE


# MODEL_NAME=$2 # "path-to-pretrained-model" stabilityai/stable-diffusion-2-base
INSTANCE_DIR=$2 # "path-to-dir-containing-your-image"
OUTPUT_DIR=$3 # "path-to-desired-output-dir" out/dreambooth/pika
CLASS_NAME=$4 # "toy"
CLASS_DIR=$5 # "data/pika/toy"

# 1st: using mask image
CUDA_VISIBLE_DEVICES=$1 python dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base"  \
  --instance_data_dir="$INSTANCE_DIR/mask" \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a foreground mask of sks $CLASS_NAME" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=25 \
  --train_text_encoder \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a photo of $CLASS_NAME" \
  --num_class_images=200 \
  --with_prior_preservation --prior_loss_weight=1.0 

# 2nd: using depth image
CUDA_VISIBLE_DEVICES=$1 python dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$OUTPUT_DIR  \
  --instance_data_dir="$INSTANCE_DIR/depth" \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a depth map of sks $CLASS_NAME" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --train_text_encoder \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a photo of $CLASS_NAME" \
  --num_class_images=200 \
  --with_prior_preservation --prior_loss_weight=1.0 

# 3rd: using normal image
CUDA_VISIBLE_DEVICES=$1 python dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$OUTPUT_DIR  \
  --instance_data_dir="$INSTANCE_DIR/normal" \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a normal map of sks $CLASS_NAME" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50 \
  --train_text_encoder \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a photo of $CLASS_NAME" \
  --num_class_images=200 \
  --with_prior_preservation --prior_loss_weight=1.0 


# 4th: use rgba image
CUDA_VISIBLE_DEVICES=$1 python dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$OUTPUT_DIR  \
  --instance_data_dir="$INSTANCE_DIR/rgba" \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a rgb photo of sks $CLASS_NAME" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=75 \
  --train_text_encoder \
  --class_data_dir=$CLASS_DIR \
  --class_prompt="a photo of $CLASS_NAME" \
  --num_class_images=200 \
  --with_prior_preservation --prior_loss_weight=1.0 

# bash dreambooth/dreambooth.sh 0 data/run/barbie_cake out_1e/barbie_cake barbie_cake images_gen/barbie_cake
# bash dreambooth/dreambooth.sh 0 data/done/metal_dragon_statue out_1e400/metal_dragon_statue metal_dragon_statue images_gen/metal_dragon_statue