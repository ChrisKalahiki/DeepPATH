# 1. Setup
## 1.1 Log into Palmetto
```
alias palmetto='ssh <your username>@login.palmetto.clemson.edu'
```

## 1.2 Start an interactive job
I recommend giving yourself a fair amount of memory and some GPUs
```
qsub -I -l select=1:ncpus=10:mem=300gb:ngpus=2:gpu_model=any:interconnect=any,walltime=2:00:00
```

## 1.3 Add modules and activate Anaconda environment
```
module load openjdk/1.8.0_222-b10-gcc/8.3.1
module load gcc/9.3.0
module load anaconda3/2020.07-gcc/8.3.1
source activate deeppath
```

## 1.4 Creating Bazel Binaries (FIRST TIME ONLY!)
I created these in my home directory. We can, however, discuss putting them in the project somewhere if that is preferred.
```
mkdir bazel-src
cd bazel-src
wget https://github.com/bazelbuild/bazel/releases/download/4.0.0/bazel-4.0.0-dist.zip

unzip bazel-4.0.0-dist.zip

env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
```
This should create Bazel binaries in `~/bazel-src/output/`

# 2. Normal Classification
It is important to note that this portion of the pipeline will involve moving to the proper directory in Palmetto.
```
cd /zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code
```

## 2.1 Pre-processing - Tiling
First, we can tile the images using the magnification (20x) and tile size of interest (512x512 px in example). The reason we tile each set separately is because the metadata has no inherent way to distinguish race of each image. This causes the sort script to require separate subfolder names to use as labels.
```
python 00_preprocessing/0b_tileLoop_deepzoom4.py  -s 512 -e 0 -j 32 -B 50 -M 20 -o out/512px_Tiled/AA "../../data/AfricanAmerican/*svs"

python 00_preprocessing/0b_tileLoop_deepzoom4.py  -s 512 -e 0 -j 32 -B 50 -M 20 -o out/512px_Tiled/C "../../data/Caucasian/*svs"
```

## 2.2 Pre-processing - Sorting
Next, we can sort the data into a train, test, and validation cohort for a 3-way classifier.
```
mkdir out/iia_sorted_3Cla
cd out/iia_sorted_3Cla
python ../../00_preprocessing/0d_SortTiles.py --SourceFolder='../512px_Tiled/' --Magnification=20.0  --MagDiffAllowed=0 --SortingOption=19 --nSplit 0 --JsonFile='../../../../data/metadata.cart.complete-iia.json' --PercentTest=15 --PercentValid=15
```

## 2.3 Pre-processing - Convert to TFRecord
Convert record into TFRecord files for each dataset
```
mkdir out/iia_TFRecord_test
mkdir out/iia_TFRecord_valid
mkdir out/iia_TFRecord_train

python 00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='out/iia_sorted_3Cla/'  --output_directory='out/iia_TFRecord_test' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test'

python 00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='out/iia_sorted_3Cla/'  --output_directory='out/iia_TFRecord_valid' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='valid'

python 00_preprocessing/TFRecord_2or3_Classes/build_image_data.py --directory='out/iia_sorted_3Cla/' --output_directory='out/iia_TFRecord_train' --train_shards=1024  --validation_shards=128 --num_threads=16
```

## 2.4 Train the 3-way Classifier
Here we are going to train our inception model on the training set we created.
```
mkdir out/iia_results

01_training/xClasses/bazel-bin/inception/imagenet_train --num_gpus=2 --batch_size=100 --train_dir='out/iia_results' --data_dir='out/iia_TFRecord_train' --ClassNumber=2 --mode='0_softmax' --NbrOfImages=712 --save_step_for_chekcpoint=200 --max_steps=2001
```

## 2.5 Validating Results
We just want to run out model on the validation set we created.
```
python 02_testing/xClasses/nc_imagenet_eval.py --checkpoint_dir='/zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code/out/iia_results/' --eval_dir='/zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code/out/iia_valid' --data_dir='/zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code/out/iia_TFRecord_valid'  --batch_size 300  --run_once --ImageSet_basename='valid_' --ClassNumber 1 --mode='0_softmax'  --TVmode='test'
```