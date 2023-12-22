# !/bin/bash
# source venv_make3d/bin/activate
which python 

nvidia-smi
nvcc --version

hostname
NUM_GPU_AVAILABLE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo "number of gpus:" $NUM_GPU_AVAILABLE

WORK_SPACE=$2 # workspace
REF_PATH=$3 # path to the images, e.g. data/nerf4/chair.png
step1=$4 # whether to use the first stage
step2=$5 # whether to use the second stage
TRAINED_MODEL_PATH=$6
CLASS_NAME=$7

FILENAME=$(basename $REF_PATH)
dataset=$(basename $(dirname $REF_PATH))
echo reconstruct $FILENAME under dataset $dataset ...

if (( ${step1} )); then
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --workspace ${WORK_SPACE} \
        --ref_path ${REF_PATH} \
        --phi_range -45 45 \
        --iters 2000 \
        --trained_model_path ${TRAINED_MODEL_PATH} \
        --class_name ${CLASS_NAME} \
        ${@:8}

    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --workspace ${WORK_SPACE} \
        --ref_path ${REF_PATH} \
        --phi_range -180 180 \
        --albedo_iters 3500 \
        --normal_iters 3000 \
        --iters 7000 \
        --trained_model_path ${TRAINED_MODEL_PATH} \
        --class_name ${CLASS_NAME} \
        --final \
        ${@:8}
fi

if (( ${step2} )); then
    CUDA_VISIBLE_DEVICES=$1 python main.py \
        --workspace ${WORK_SPACE} \
        --ref_path ${REF_PATH} \
        --phi_range -180 180 \
        --refine \
        --iters 7000 \
        --final \
        --trained_model_path ${TRAINED_MODEL_PATH} \
        --class_name ${CLASS_NAME} \
        --process_mask \
        --save_mesh \
        --mesh_pcd \
        ${@:8}
fi

# bash scripts/run.sh 0 armchair_1e data/run/armchair/rgba/rgba.png 1 0 out_1e/armchair armchair