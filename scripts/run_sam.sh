hostname
NUM_GPU_AVAILABLE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo "number of gpus:" $NUM_GPU_AVAILABLE

# INPUT=$2 # Path to either a single input image or folder of images.
# OUTPUT=$3 # Path of outcome folder, don't need exits in advance
REFPATH=$2 # Path for reference image
WORKSPACE=$3 # Workspace for reconstruction
# INPUT="mask/$WORKSPACE/rgbs"
# OUTPUT="mask/$WORKSPACE/sam_masks"

# prepare rgb files for SAM
CUDA_VISIBLE_DEVICES=$1 python mask/mask_utils.py \
    --ref_path ${REFPATH} \
    --workspace ${WORKSPACE} \
    ${@:4}

# generate masks by SAM for whole image
# CUDA_VISIBLE_DEVICES=$1 python mask/amg.py \
#     --checkpoint "mask/sam_vit_h_4b8939.pth" \
#     --model-type "default" \
#     --input ${INPUT} \
#     --output ${OUTPUT} \
#     ${@:4}

# # process masks
# CUDA_VISIBLE_DEVICES=$1 python mask/mask_utils.py \
#     --ref_path ${REFPATH} \
#     --workspace ${WORKSPACE} \
#     --sam_output ${OUTPUT} \
#     ${@:4}
