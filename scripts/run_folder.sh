topdir=$1 # path to the images dir, e.g. data/nerf4
step1=$2 # whether to use the first stage
step2=$3 # whether to use the second stage

echo "run pipeline in $topdir...."

for i in $topdir/*; do
    echo "$i" # image/nerf4/chair
    [ -d "$i" ] && echo "$i exists."
    ref_path="$i/rgba/rgba.png"
    filename=$(basename $i)
    trained_model_path="out/$filename"
    class_name="$filename"
    workspace=$(basename $i)
    if [ -d "$trained_model_path" ]; then
        bash scripts/run.sh 0 $workspace $ref_path $step1 $step2 $trained_model_path $class_name ${@:4}
    fi
done

# bash scripts/run_folder.sh data/nerf4 1 0