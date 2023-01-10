# VanillaGradientDescent
This was a winter break learning project for me to get introduced into the world of AI. From scratch I build a neural network that uses mini-batch gradient descent. I then applied it to recognize hand-written digits from the sklearn database.

I simulated the network with the sklearn dataset, training on 1600 samples and testing with 197. It averages around 94% accuracy, although the testing samples are small. It runs slow on the MNIST dataset so I didn't bother examining it's success there. Examining other similar projects online I have suspicions as to why its so slow. I believe I was not space efficient with storing activations, and that I used extra loops that were unnecessary. Other than that my approach is quite similar to others. I wanted to improve this but alas school has started again so I have diverted my focus there. Additionally, this wasn't meant to be a notable professional accomplishment, rather just a chance for me to get my hands dirty with neural networks and build something from the ground up. It was a lot of fun. 

The network learns, and does well at classification so I am taking that as a win (for now). Below I have compiled the resources I used while learning and preparing for this project.


https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
http://colah.github.io/posts/2015-08-Backprop/
http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm
<br> Not to mention numerous stack overflow threads reading about types of gradient descent, the cost surfaces, and what to set for the parameters

( An identical project I found at the end to compare efficiency) https://eng.libretexts.org/Bookshelves/Computer_Science/Applied_Programming/Book%3A_Neural_Networks_and_Deep_Learning_(Nielsen)/01%3A_Using_neural_nets_to_recognize_handwritten_digits/1.07%3A_Implementing_our_network_to_classify_digits


# Dependencies
numpy
sklearn.datasets
