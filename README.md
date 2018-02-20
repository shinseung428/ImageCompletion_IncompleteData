# Learning to complete images using only incomplete data  

Most of the image completion models use fully-observed samples to train the network. However, as it's stated in the ambientGAN paper, obtaining high resolution samples can be very expensive or impractical for some applications.  

The model in this project combines two ideas from following works:  
* AmbientGAN  
* Globally and Locally Consistent Image Completion  

The AmbientGAN model enables training a generative model directly from noisy or incomplete samples. The generator in the model successfully predicts samples from the true distribution with the use of a measurement function.   

On the other hand, the model in Globally and Locally Consistent Image Completion uses fully-observed to train the completion network. The network uses mse loss and discriminator loss to train the network, and in this process, fully-observed images are used.  

By combining two ideas presented in ambientGAN and GLCIC paper, the network presented in this project learns to fill incomplete regions in the images using only incomplete data (e.g. images randomly blocked by p x p patch).  


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

## Conclusion  


## Related Projects  
* [AmbientGAN](https://openreview.net/forum?id=Hy7fDog0b)
* [Globally and Locally Consistent Image Completion](http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/en/)  

