# 1. Normal Classification
## 1.1 Pre-processing - Tiling
First, we can tile the images using the magnification (20x) and tile size of interest (512x512 px in example)
python 00_preprocessing/0b_tileLoop_deepzoom4.py  -s 512 -e 0 -j 32 -B 50 -M 20 -o 512px_Tiled "../../data/*/*svs"

## 1.2 Pre-processing - Sorting
Next, we can sort the data into a train, test, and validation cohort for a 3-way classifier.
```
mkdir iia_sorted_3Cla
cd iia_sorted_3Cla
python ../00_preprocessing/0d_SortTiles.py --SourceFolder='../512px_Tiled/' --Magnification=20.0  --MagDiffAllowed=0 --SortingOption=10 --nSplit 0 --JsonFile='../../../metadata.cart.complete-iia.json' --PercentTest=15 --PercentValid=15
```

## 1.3 Pre-processing - Convert to TFRecord
Convert record into TFRecord files for each dataset
```
mkdir r1_TFRecord_test
mkdir r1_TFRecord_valid
mkdir r1_TFRecord_train

python 00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='r1_sorted_3Cla/'  --output_directory='r1_TFRecord_test' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='test'

python 00_preprocessing/TFRecord_2or3_Classes/build_TF_test.py --directory='r1_sorted_3Cla/'  --output_directory='r1_TFRecord_valid' --num_threads=1 --one_FT_per_Tile=False --ImageSet_basename='valid'

python 00_preprocessing/TFRecord_2or3_Classes/build_image_data.py --directory='r1_sorted_3Cla/' --output_directory='r1_TFRecord_train' --train_shards=1024  --validation_shards=128 --num_threads=16
```

## 1.4 Train the 3-way Classifier
```
mkdir r1_results

01_training/xClasses/bazel-bin/inception/imagenet_train --num_gpus=2 --batch_size=100 --train_dir='r1_results' --data_dir='r1_TFRecord_train' --ClassNumber=2 --mode='0_softmax' --NbrOfImages=712 --save_step_for_chekcpoint=200 --max_steps=2001
```
