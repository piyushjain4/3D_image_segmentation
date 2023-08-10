# 3D_image_segmentation_via_deep_CNN

### Description:

- These models were trained for the [kaggle compitition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/overview) and won a **BRONZE** medal.
- Given a fragment, this segmentation model detects the ink.
- The output of the model is a binary mask where the 1 represents the presence of ink and 0 for no ink.
- This above mask is then converted into the RLE(Run Length Encodings).


   <img src="https://github.com/Vishak-Bhat30/3D_image_segmentation/assets/102585626/e62e7bd6-71de-43b0-9164-effdec6dd51c" alt="RLE" width="300" />
The following example has the letter "E" it's RLE encoding is: "11 7 20 1 23 1 26 1 29 1 32 1 35 1 38 1 44 1"

 
### Dataset:
- The data is taken from the [kaggle compitition](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data).
<img src="https://github.com/Vishak-Bhat30/3D_image_segmentation/assets/102585626/c3b0965c-7bbf-4144-9a1d-95532ede7b88" width="200" />

- The dataset is 3d x-ray scans of detached fragments of ancient papyrus scrolls.
- The training data had 3 fragments.
- It had the slices from the 3d x-ray surface volume. Each file contains a greyscale slice in the z-direction. Each fragment contains       65 slices. Combined this image stack gives us width * height * 65 number of voxels per fragment.


  <img src="https://github.com/Vishak-Bhat30/3D_image_segmentation/assets/102585626/1e48b650-c888-45b4-8930-ab69f8b69b06" alt="The 65 channels" width="400" />

  
 The 65 slices of the first fragment
 
- The inklables were given which was a binary mask which showed 1 for the presense of the ink.


  <img src="https://github.com/Vishak-Bhat30/3D_image_segmentation/assets/102585626/8e0f1b70-be98-4a6d-8668-78a551a83545" alt="labels" width="200" />


- Further there was even the mask of the fragment which basically shows where the data is present in the fragment.


  <img src="https://github.com/Vishak-Bhat30/3D_image_segmentation/assets/102585626/47fa9262-e9c0-4b9d-ba47-157053117633" alt="The 65 channels" width="200" />
  
### Contents

- Training_notebooks: Contains the notebooks used for training. Much more description about them is given in the training readme.
- Inference_notebooks: Contains the notebooks used for Inference. Much more description about them is given in the Inference readme.
- tiles_extracted: This is the notebook that reduces the dataset by merging the volume scans with the mask. This removes the 
       unecessary tiles for training the model


### Training

- Trained multiple U-Net models using the segmentation-models-pytorch.
- Trained models with the backbone 
    1) mit_b2 , mitb3 , mit_b4 , mit_b5
    2) VGG19
    3) resnet50 resnet34
    4) efficientnet_b3
    5) resnest
    6) regnety 
    etc
    
- Few of them can be found in the repo.
- Trained a model using stratified K fold technique
- During the training the fragments were broken into the sizes of 224 by 224 and kept a stride of 224//4
- trained for 15 epochs by keeping one of the 3 fragments as the cross validation 
- The calculated metric was fbeta score keeping the beta value 0.5
- Was getting the CV score of single model around 63
- used the weighted loss of dice loss and the Tversky loss and BCE loss



<img src="https://github.com/Vishak-Bhat30/3D_image_segmentation/assets/102585626/be5836b4-6fef-45c1-8ab9-b5371fbc354b" width="1000" />

fig: the first image is the actual mask, the second one is the predicted mask, third one is the predicted mask after applying a threshhold 0.4


### Inference Time

#### Ensembles

- Did the ensembles of different models trained, tried average ensemble.
- Ensembling the models increased the fbeta score from 65 to 74.
- Tried weighted ensembles tooo

#### TTA

- Performed TTA(Test Time Augmentations) during the final inference.
- This included the rotation of the image during the inference.(did 90,180,270,360 degree roatations)
        
          x = [x, *[tc.rot90(x, k=i, dims=(-2, -1)) for i in range(1, 4)]]
          x = tc.cat(x, dim=0)
            
- after predicting on these augmentations, again rotated the output then took the average of the outputs

          x=[tc.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
          x=tc.stack(x,dim=0)


#### 8 Channels average score

- Out of the 65 channels, the center channels contained the information.
- So took the center 8 channels and made them into group of 3 because the model trained took only 3 channels in input
- The groups of 3 channels were (1,2,3) , (4,5,6) , (6,7,8)
- So this resulted in output of 3 masks which was later averaged to get the final mask



