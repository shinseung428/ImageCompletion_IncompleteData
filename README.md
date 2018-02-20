# Learning to complete images using only noisy observations  

Most of the image reconstruction models uses fully-observed samples to train the network. However, obtaining high resolution samples can be very expensive or impractical for some applications. 

The model in this project can be thought as a combination of two ideas:  
* ambientGAN  
* Globally and Locally Consistent Image Completion  

The ambientGAN model enables training a generative model directly from noisy or incomplete samples.  

The model in Globally and Locally Consistent Image Completion paper successfully uses fully-observed data and trains the completion network to fill incomplete regions in the image. 

By using two ideas presented in ambientGAN and GLCIC, the network presented in this project learns to fill incomplete regions in the images without the use of fully-observed data. 


## Network



