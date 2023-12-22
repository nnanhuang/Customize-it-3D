examples=(
    'data/run/cat_statue'
    # 'data/run/armchair'
    # 'data/run/metal_dragon_statue'
    # 'data/realfusion15/stone_dragon_statue'
    # 'data/done/teddy_bear'
)

for i in "${examples[@]}"; do
    echo "$i" # image/nerf4/chair
    [ -d "$i" ] && echo "$i exists."

    ref_path="$i/rgba.png"

    CUDA_VISIBLE_DEVICES=$1 python preprocess_image.py \
        --path $ref_path
done

# use: bash scripts/preprocess_list.sh 0