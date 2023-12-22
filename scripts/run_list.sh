step1=$1 # whether to use the first stage
step2=$2 # whether to use the second stage

examples=(
    # 'data/realfusion15/teddy_bear'
    # 'data/run/armchair'
    # 'data/realfusion15/cat_statue'
    'data/run/metal_dragon_statue'
    'data/done/barbie_cake'
    # 'data/realfusion15/stone_dragon_statue'
)

for i in "${examples[@]}"; do
    echo "$i"
    [ -d "$i" ] && echo "$i exists."
    ref_path="$i/rgba.png"
    workspace=$(basename $i)
    trained_model_path="out_5e400/$workspace"
    class_name="$workspace"
    if [ -d "$trained_model_path" ]; then
        bash scripts/run.sh 0 $workspace $ref_path $step1 $step2 $trained_model_path $class_name ${@:4}
    fi
done

# usage: bash scripts/run_list.sh 1 0