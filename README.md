# Learning to complete images using incomplete data  

Most of the image completion and generative models require fully-observed samples to train the network. However, as it's stated in the ambientGAN paper, obtaining high resolution samples can be very expensive or impractical for some applications.  

The model in this project combines ideas from the following works:  
* AmbientGAN  
* Globally and Locally Consistent Image Completion  

The AmbientGAN model enables training a generative model directly from noisy or incomplete samples. The generator in the model successfully predicts samples from the true distribution with the use of a measurement function.   

On the other hand, the model in Globally and Locally Consistent Image Completion uses fully-observed samples to train the network. The completion network first uses mse loss to pre-train the weights and further uses a discriminator loss to fully train the network.  

By combining the ideas presented in ambientGAN and GLCIC paper, the network presented in this project learns to fill incomplete regions using only incomplete data (e.g. images randomly blocked by 28 x 28 patch).  


## Network
![Alt text](images/network.png?raw=true "network")  


## Method
Let's assume that we have incomplete samples and we know the type of noise added to the samples.  
Instead of using a random latent vector as an input, the completion network gets masked image as an input. Assuming that the completion network successfully generates the masked region, the generated patch is combined together with the input image using the mask information X<sub>g</sub>.  

Next, the completed image X<sub>g</sub> is fed into the measurement function. As described in the AmbientGAN paper, the measurement function tries to simulate the random measurements on the generated objects X<sub>g</sub>. This is possible since we know the type of noise added to the fully-observed images. We can create a measurement function that could simulate the noise added to the image.  

The resulting image Y<sub>g</sub> after the measurement function and the incomplete samples Y<sub>r</sub> are then passed on to the discriminator that distinguishes real measurements from the fake measurements. As the completion network and the discriminator network are trained adversarially, the completion network learns to generate patches that goes well with the incomplete samples.    

## Dataset  
[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset was used in this project. To create incomplete dataset, the original CelebA image was center cropped by 32x32 patch and was resized to 64x64. Then 28x28 patch was added randomly to the image. The blocked regions were filled with ones.  

![Alt text](images/dataset.jpg?raw=true "dataset")  

## Training  
```
$ python train.py 
```

To continue training  
```
$ python train.py --continue_training=True
```


## Results  
![Alt text](images/block_patch_res_1.gif?raw=true "res_1")  
![Alt text](images/block_patch_res_2.gif?raw=true "res_2")  

## Conclusion  
The resulting images show that the presented model learns to fill incompelte regions. There are some images filled with artifacts and inconsistent colors, but in general the completion network learns to generate patches that goes well with the input image.  


## Possible Improvements  
* Post-processing to fix color inconsistency  


## Related Projects  
* [AmbientGAN](https://openreview.net/forum?id=Hy7fDog0b)
* [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/)  

