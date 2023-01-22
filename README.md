# VanillaGradientDescent
This was a winter break learning project for me to get introduced into the world of AI. I built a neural network from scratch, that uses mini-batch gradient descent. I then applied it to recognize hand-written digits from the sklearn database.

I simulated the network with the sklearn dataset, training on 1600 samples and testing with 197. From the personal tests I ran, it averages around 96% accuracy sometimes as high as 98%. This project, as stated previously, was done from scratch as an example to demonstrate, test, and implement my learning. As such, the first version while functional was sloppy. It performed the same, accuracy-wise, just substantially slower. I placed this version in the 'original' folder. If anyone wants to see a slow and memory-intensive manner of doing things its there! The 'network_optimized' and 'sklearn_network_optimized' files refer to my revised version. After reading other networks online I realized I could collapse a lot of for loops and let numpy handle more than I needed to so I rewrote the backpropogation and initialization functions, as well as restructured the network a tad bit, and it performs much faster now. The sklearn script is what runs the network on the sklearn dataset and evaluates it. 

Note that the network script itself is not at all unique to the sklearn or even digit recognition at all, it is just a network that works with vectors of any dimension specified in the initialization of the object. This is because I intend to use this in later projects and want it to be re-purposable. It's also more elegant this way I feel. 

For those who want to see what I read to learn the resources are below: <br>
https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
http://colah.github.io/posts/2015-08-Backprop/
http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm
<br> Not to mention numerous stack overflow threads reading about types of gradient descent, the cost surfaces, and what to set for the parameters

( An identical project I found at the end to compare efficiency) https://eng.libretexts.org/Bookshelves/Computer_Science/Applied_Programming/Book%3A_Neural_Networks_and_Deep_Learning_(Nielsen)/01%3A_Using_neural_nets_to_recognize_handwritten_digits/1.07%3A_Implementing_our_network_to_classify_digits


# Dependencies
+ numpy <br>
+ sklearn.datasets

# Running It
Just run the main file sklearn_optimized after importing the two dependencies into your environment. You can tweak the training_amount, epochs, minibatch_size, and learning_rate parameters to see how it affects the performance of the network accuracy-wise. You can also add more or less neurons and layers in the initialization of the network. Note that the initialization must contain an input and output layer of non-zero amount.
