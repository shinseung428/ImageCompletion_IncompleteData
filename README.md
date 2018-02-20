# Learning to complete images using only noisy observations  

Most of the image completion models use fully-observed samples to train the network. However, as it's stated in the ambientGAN paper, obtaining high resolution samples can be very expensive or impractical for some applications.  

The model in this project combines two ideas from the following papers:  
* ambientGAN  
* Globally and Locally Consistent Image Completion  

The ambientGAN model enables training a generative model directly from noisy or incomplete samples. The generator successfully predicts samples from the true distribution with the use of a measurement function. 

The model in Globally and Locally Consistent Image Completion paper successfully uses fully-observed data and trains the completion network to fill incomplete regions in the image. The completion network is first trained using an mse loss, then it is further trained using the discriminator loss.  

By using two ideas presented in ambientGAN and GLCIC, the network presented in this project learns to fill incomplete regions in the images using incomplete data (e.g. images randomly blocked by p x p patch). 


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


