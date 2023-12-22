topdir=$1 # path to the images dir, e.g. data/nerf4
# INSTANCE_DIR=$3 # "path-to-dir-containing-your-image"
# OUTPUT_DIR=$5 # "path-to-desired-output-dir" out/pika
# CLASS_NAME=$6 # "toy"
# CLASS_DIR=$7 # "data/pika/toy"
echo "fine-tune with images in $topdir...."

for i in $topdir/*; do
    echo "$i" # image/nerf4/chair
    [ -d "$i" ] && echo "$i exists."
    class_name=$(basename $i)
    output_dir="out/$class_name"
    class_dir="images_gen/$class_name"
    if [ -d "$output_dir" ]; then
            echo " '$output_dir' exists,contiue "
        continue
    fi
    bash dreambooth/dreambooth.sh 0 $i $output_dir $class_name $class_dir ${@:2}
done

# bash scripts/run_dreambooth_folder.sh data/run