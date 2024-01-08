# Stabilizing Spiking Neuron Training

This is the official repository of the [Stabilizing Spiking Neuron Training](https://arxiv.org/abs/2202.00282) 
article, submitted to IEEE.

Run the following commands to reproduce the results:

- ```training.py``` for each individual experiments;
- ```plots.py``` to plot the results and the figures.

Figure 1 can be generated running ```plot_sgs.py```. To generate the experiments
for Figure 2, run as one line

```
python training.py with task_name=##x## net_name=##y## 
     epochs=None steps_per_epoch=None batch_size=None stack=None n_neurons=None 
     lr=##z## comments=embproj_nogradreset_dropout:.3_timerepeat:2_##w## 
```


where ```##x##``` should be changed by one of the following three tasks ```wordptb```, ```heidelberg```, ```sl_mnist```,
```##y##``` by one of the following three networks ```maLIF``` for the LIF, 
```maLSNN``` for the ALIF and ```spikingLSTM```, 
```##z##``` by one of the learning rates in the list ```[1e-5, 3.16e-5, 1e-4, 3.16e-4, 1e-3, 3.16e-3, 1e-2]```, 
and ```##w##``` is used to specify the SG shape, and should be changed by one of the following 
```cappedskippseudod``` for the rectangular, ```originalpseudod``` for the triangular,
```exponentialpseudod``` for the exponential,```gaussianpseudod``` for the gaussian, 
```sigmoidalpseudod``` for the derivative of the sigmoid, 
```fastsigmoidpseudod``` for the derivative of the fast sigmoid.

