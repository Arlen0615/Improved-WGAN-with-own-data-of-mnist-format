# Improved-WGAN-with-own-data-of-mnist-format

This project is implement Improved WGAN, Conditional Improved WGAN and Deep Conditional Improved WGAN with tensorflow
Use [Convert-own-data-to-MNIST-format](https://github.com/Arlen0615/Convert-own-data-to-MNIST-format)  tool, you can test any own data with Improved WGAN

First, I need thanks [Agustinus Kristiadi](https://github.com/wiseodd), I reference his code so much.

wgan_gp_tensorflow.py is modified from [improved_wasserstein_gan](https://github.com/wiseodd/generative-models/tree/master/GAN/improved_wasserstein_gan)

cwgan_gp_tensorflow.py is combined with [conditional_gan](https://github.com/wiseodd/generative-models/tree/master/GAN/conditional_gan) and [improved_wasserstein_gan](https://github.com/wiseodd/generative-models/tree/master/GAN/improved_wasserstein_gan)

dcwgan_gp_tensorflow.py is inherit with cwgan_gp_tensorflow.py and change generator and discrimintor to CNN model.

All of the model is re-design for meet own data with RGB images.

***Note:***  
Please make sure your machine have enough memory, or you could reduce train effort in my code. 
