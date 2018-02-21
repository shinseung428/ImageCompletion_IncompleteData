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
The resulting images show that the presented model fills incompelte regioms in the images. There are some artifacts and unnatural textures in the filled regions, but in general the completion network learns to generate patches that goes well with the input image.

## Related Projects  
* [AmbientGAN](https://openreview.net/forum?id=Hy7fDog0b)
* [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/)  

