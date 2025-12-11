## Download Dataset

Download the Cityscapes dataset [here](https://www.cityscapes-dataset.com/downloads/):
- gtFine_trainvaltest.zip (241MB)
  - fine annotations for train and val sets (3475 annotated images) and dummy annotations (ignore regions) for the test set (1525 images)
- gtCoarse.zip (1.3GB)
  - coarse annotations for train and val set (3475 annotated images) and train_extra (19998 annotated images)
- leftImg8bit_trainvaltest.zip (11GB)
  - left 8-bit images - train, val, and test sets (5000 images)
- leftImg8bit_trainextra.zip (44GB)
  - left 8-bit images - trainextra set (19998 images, note that the image "troisdorf_000000_000073_leftImg8bit.png" is corrupt/black)

After downloading extract the zip files and move them into `data/City Scapes` directory

Structure of dataset directory for City Scapes should look like this:

```
data/
└── City Scapes/
    ├── gtCoarse/...
    ├── gtFine_trainvaltest/...
    ├── leftImg8bit_trainvaltest/...
    └── leftImg8bit_trainextra/...
```

### Preprocessing

Before training, you must generate the `trainId` label images. The default Cityscapes annotations contain many classes that are not used for training. The following script converts these annotations into the standard 19 classes (TrainIDs) and saves them as `_labelTrainIds.png` files, which are required for the dataloader.

Run the following commands:

```bash
# Set environment variable to the dataset directory
export CITYSCAPES_DATASET="./data/cityscapes"

# Generate trainId label images
python -m cityscapesscripts.preparation.createTrainIdLabelImgs