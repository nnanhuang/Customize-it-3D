topdir=$2

echo "preprocess images in $topdir...."

for i in $topdir/*; do
    echo "$i" # image/nerf4/chair
    [ -d "$i" ] && echo "$i exists."

    ref_path="$i/rgba.png"

    CUDA_VISIBLE_DEVICES=$1 python preprocess_image.py \
        --path $ref_path
done
# use: bash scripts/preprocess_folder.sh 0 data/run