# made

PyTorch implementation of the MADE model ([Masked Autoencoder for Distribution Estimation](https://arxiv.org/abs/1502.03509)).


```python
from made import MADE

model = MADE(input_size=784, hidden_sizes=[500], output_size=784)
```


## MNIST Training

First download the binarized MNIST dataset

```
wget -O .data/binarized_mnist.npz https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz 
```

Run the training script. The following script create a model of hidden size (500, 1000), train for 5 epoch, and save the model weight to `.data/mnist_made.pt`
```
python train.py -s 500,1000 -e 5 -f .data/mnist_made.pt
```

Create image samples
```
python sample.py -s 500,1000 -f .data/mnist_made.pt
```

Sample image generated by MADE model

![image](https://github.com/stkao05/made/assets/1556390/cde50130-a191-400d-8bce-10d41aa385dd)
