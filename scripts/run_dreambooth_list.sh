# INSTANCE_DIR=$3 # "path-to-dir-containing-your-image"
# OUTPUT_DIR=$5 # "path-to-desired-output-dir"
# CLASS_NAME=$6 # "toy"
# CLASS_DIR=$7 # "data/pika/toy"
out_name=$1
examples=(
    # 'data/realfusion15/cat_statue'
    'data/run/armchair'
    'data/run/metal_dragon_statue'
    # 'data/realfusion15/stone_dragon_statue'
    # 'data/realfusion15/teddy_bear'
)

for i in "${examples[@]}"; do
    echo "$i" # image/nerf4/chair
    [ -d "$i" ] && echo "$i exists."
    class_name=$(basename $i)
    output_dir="$out_name/$class_name"
    class_dir="images_gen/$class_name"
    if [ -d "$output_dir" ]; then
            echo " '$output_dir' exists,contiue "
        continue
    fi
    bash dreambooth/dreambooth.sh 0 stabilityai/stable-diffusion-2-base $i $output_dir $class_name $class_dir 200 ${@:6}
done

# bash scripts/run_dreambooth_list.sh out_nomask