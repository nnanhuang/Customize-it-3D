topdir=$1 # path to the images dir, e.g. data/nerf4
step1=$2 # whether to use the first stage
step2=$3 # whether to use the second stage

# 1. preprocess ~1min
bash scripts/preprocess_folder.sh 0 $topdir
# 2. fine-tune dream booth ~10min
bash scripts/run_dreambooth_folder.sh $topdir
# 3. train customize-it-3D ~2h
bash scripts/run_folder.sh $topdir $step1 $step2

# use: bash scripts/run_full_folder.sh data/run 1 0