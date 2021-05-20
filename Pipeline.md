# 1. Setup
## 1.1 Log into Palmetto
```
ssh <your username>@login.palmetto.clemson.edu
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

# 2. Classification
It is important to note that this portion of the pipeline will involve moving to the proper directory in Palmetto.
```
cd /zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code
```

## 2.1 Pre-processing - Tiling
First, we can tile the images using the magnification (20x) and tile size of interest (512x512 px in example). The reason we tile each set separately is because the metadata has no inherent way to distinguish race of each image. This causes the sort script to require separate subfolder names to use as labels.
```
python 00_preprocessing/0b_tileLoop_deepzoom4.py -s 512 -e 0 -j 32 -B 50 -M 20 -o out/512px_Tiled/AA "../../data/AfricanAmerican/*svs"

python 00_preprocessing/0b_tileLoop_deepzoom4.py -s 512 -e 0 -j 32 -B 50 -M 20 -o out/512px_Tiled/C "../../data/Caucasian/*svs"

python 00_preprocessing/0b_tileLoop_deepzoom4.py -s 512 -e 0 -j 32 -B 50 -M 20 -o out/512px_Tiled/H "../../data/Hispanic/*svs"
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

python 01_training/xClasses/bazel-bin/inception/imagenet_train --num_gpus=2 --batch_size=100 --train_dir='out/iia_results' --data_dir='out/iia_TFRecord_train' --ClassNumber=3 --mode='0_softmax' --NbrOfImages=712 --save_step_for_chekcpoint=200 --max_steps=2001
```

## 2.5 Validating Results
We just want to run our trained model on the test and validation sets we created.
```
python 02_testing/xClasses/nc_imagenet_eval.py --checkpoint_dir='/zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code/out/iia_results/' --eval_dir='/zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code/' --data_dir='/zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code/out/iia_TFRecord_test'  --batch_size 300  --run_once --ImageSet_basename='test_' --ClassNumber 3 --mode='0_softmax'  --TVmode='test'

python 02_testing/xClasses/nc_imagenet_eval.py --checkpoint_dir='/zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code/out/iia_results/' --eval_dir='/zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code/' --data_dir='/zfs/dzrptlab/breastcancer/DeepPATH/DeepPATH_code/out/iia_TFRecord_valid'  --batch_size 300  --run_once --ImageSet_basename='valid_' --ClassNumber 3 --mode='0_softmax'  --TVmode='valid'
```

# 3 Post-Processing
Generate heat-maps per slides overlaid on original slide (all test slides in a given folder; code not optimized and slow):

## 3.1 Heatmap with Overlay (Slow method)
```
python 03_postprocessing/0f_HeatMap_nClasses.py  --image_file 'out/iia_sorted_3Cla' --tiles_overlap 0 --output_dir './out/Heatmap_out' --tiles_stats 'out_filename_Stats.txt' --resample_factor 10 --slide_filter 'test_' --filter_tile '' --Cmap '' --tiles_size 512
```

## 3.2 Heatmap with no Overlay (Fast method)
Generate heat-maps with no overlay (fast)
```
python 03_postprocessing/0g_HeatMap_MultiChannels.py --tiles_overlap=0 --tiles_size=512 --output_dir='out/CMap_output' --tiles_stats='out_filename_Stats.txt' --Classes='1,2,3' --slide_filter=''
```

## 3.3 Confidence Interval Information
To also get confidence intervals (Bootstrap technique), use this code:
```
python 03_postprocessing/0h_ROC_MultiOutput_BootStrap.py  --file_stats out_filename_Stats.txt  --output_dir out/ROC_out --labels_names labelref_r1.txt --ref_stats ''
```

# 4 Image Upload Pipeline
When cropping, we want to keep the filenames the same. If possible, keep them in folders with race as that is not metadata kept on the SVS file.
```
scp -r [path to local images] username@xfer-01.palmetto.clemson.edu:/zfs/dzrptlab/breastcancer/data_cropped/[path to folder]
```

# 5 HistomicsUI Web Application
For the HistomicsUI Web Applciation, we have been using the Docker image provided by DigitalSlideArchive on GitHub. The URL for installing and setting up the Docker image can be found at the below URL.
URL: https://github.com/DigitalSlideArchive/digital_slide_archive/blob/master/ansible/README.rst
It should be noted that the instructions are designed for a Debian-based Linux installation. For Windows and MacOS, be sure to install Docker, Python, pip, and git beforehand.

# 6 Cropping SVS Images
For cropping the SVS images, we plan to use Dr. Iuricich's roi_necrosis repository. It is listed at the URL below.
URL: https://github.com/IuricichF/roi_necrosis
For the image cropping to work, the names of the annotations must be identical to the names of the original .SVS files. 
Once you are finished annotating the slide, you will need to download the annotation. Visual instructions can be found in the README.md file in the roi_necrosis repository.
