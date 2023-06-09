- Data curation - https://www.tensorflow.org/tutorials/images/segmentation
  - Explore data
  - How to split data
  - Explore segmentation techniques 2D images
  - Augmentation:
    - Random cropping, scaling, resize, padding
- Check different architectures (eg Unet)
  - Unet
  - ResNet
  - VGG
  - Inception
  - LinkNet
  - Mask R-CNN
  - FPN
- Search for transfer learning for the encoder (MobileNetV2)

- Task that we have done
  - extra test stage 1 data from giving excel and join all masks together as one final combined mask
  - explore the data and use test 1 data set as validation set (Nucleus Density, color, brightness and dimension)
  - check different pre-processing method (scale, scale with padding, padding, cropping 'sliding window with overlap, average value will be chosen at overlapping area for mask rejoining')
  - gray scale (invert some images to make all images have black background), normalization to gray scale, and increase the contrast ration
  - two approach are choosen (UNET and transfer MobileNetV2) - MeanIoU are to be compared to choose to explore further

- Possible options that we can try after choosing the model
  - Adam optimizer compared to default SGD optimizer
  - Dice coeficient or binary cross-entropy
  - Colored picture (5 different images types)
  - Contract ratio
  - Standarize or normalize image (I dont think they make any differents)
  - learning rate callback (Cosine Learning rate decay or Exponential decay)
  - Correct data set
  - Extra dataset
  - Fill holes (can be helpful, pre-&post-procesing)
  - Mosaics (have not checked yet, pre-procesing)
  - DETECTION_MASK_THRESHOLD (seems to help if model cannot predict light shadow well)
  - Augmentation (very helpful since we really have small dataset)
  - Scaled image (just to back up crop method, pre-procesing)
  - Combined predictions on actual image and horizontally flipped image (it is said helpful on the forum, post-processing)
  - Dilating and then eroding individual masks (opencv) (maybe??? pre-procesing)

- Augmentation
  - Relied heavily on image augmentation due to small training set:
  - Random horizontal or vertical flips
  - Random 90 or -90 degrees rotation
  - Random rotations in the range of (-15, 15) degrees
  - Random cropping of bigger images and masks to dedicated dimension
  - Random scaling of image and mask scaling in the range (0.5, 2.0)
