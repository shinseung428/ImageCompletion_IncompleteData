# Tensorflow implementation of AmbientGAN

### folder setting
```
-data
  -img_align_celeba
    -img1.jpg
    -img2.jpg
    -...
```
## Measurement models
* block_pixel : Each pixel is independently set to zero with probability p.  
* block_patch : A randomly chosen k × k patch is set to zero.  
* keep_patch : All pixels outside a randomly chosen k × k patch are set to zero.  
* conv_noise: k sized gaussian kernel is convolved and noise is added from the distribution Θ ∼ pθ.   

## Requirements
* python 2.7
* Tensorflow 1.4
* numpy
* cv2 (to save image tile)


## Training
```
$ python train.py --measurement=block_pixel
```

To continue training  
```
$ python train.py --measurement=block_pixel --continue_training=True
```


## Block-Pixels

***Trained CelebA images***  
![Alt text](images/blockpixels_train.jpg?raw=true "blockpixels celeba")  
***Results***  
![Alt text](images/blockpixels_result.jpg?raw=true "blockpixels result")  
![Alt text](images/blockpixels_result.gif?raw=true "blockpixels result gif")  


## Block-Patch

***Trained CelebA images***  
![Alt text](images/blockpatch_train.jpg?raw=true "blockpatch celeba")  
***Results***  
![Alt text](images/blockpatch_result.jpg?raw=true "blockpatch result")  
![Alt text](images/blockpatch_result.gif?raw=true "blockpatch result gif")  

## Keep-Patch

***Trained CelebA images***  
![Alt text](images/keeppatch_train.jpg?raw=true "keeppatch celeba")  
***Results***  
![Alt text](images/keeppatch_result.jpg?raw=true "keeppatch result")  
![Alt text](images/keeppatch_result.gif?raw=true "keeppatch result gif")  


## Convolve+Noise Result

## ToDo
* Test Convolve+Noise Result



