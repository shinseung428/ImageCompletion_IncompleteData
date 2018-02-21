# Learning to complete images using only incomplete data  

Most of the image completion and generative models require fully-observed samples to train the network. However, as it's stated in the ambientGAN paper, obtaining high resolution samples can be very expensive or impractical for some applications.  

The model in this project combines ideas from the following works:  
* AmbientGAN  
* Globally and Locally Consistent Image Completion  

The AmbientGAN model enables training a generative model directly from noisy or incomplete samples. The generator in the model successfully predicts samples from the true distribution with the use of a measurement function.   

On the other hand, the model in Globally and Locally Consistent Image Completion uses fully-observed samples to train the network. The completion network first uses mse loss to pre-train the weights and further uses a discriminator loss to fully train the network.  

By combining ideas presented in ambientGAN and GLCIC paper, the network presented in this project learns to fill incomplete regions in the images using only incomplete data (e.g. images randomly blocked by p x p patch).  


## Network
![Alt text](images/network.png?raw=true "network")  


## Method
Let's assume that we have incomplete samples and we know the type of noise added to the samples.  
Instead of using a random latent vector as an input, the completion network gets masked image as an input. Assuming that the completion network successfully generates the masked region, the generated patch is combined together with the input image using the mask information X<sub>g</sub>. Next, the completed image is fed into the measurement function. As described in the AmbientGAN paper, the measurement function tries to simulate the random measurements on the generated objects X<sub>g</sub>. This is possible since we know the type of noise added to the fully-observed images. We can create a measurement function that could simulate the noise added to the image. The resulting image given by the measurement function is then passed on to the discriminator that distinguishes real measurements from the fake measurements.  

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
The resulting images show that the presented model fills incompelte regions in the images. There are some images showing artifacts and unnatural textures in the filled regions, but in general the completion network learns to generate patches that goes well with the input image.  


## Possible Improvements  
* Post-processing to fix texture issue  


## Related Projects  
* [AmbientGAN](https://openreview.net/forum?id=Hy7fDog0b)
* [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/)  

