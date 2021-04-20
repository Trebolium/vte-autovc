## AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss

This repository features an adaptation of the AutoVC framework presented in [Zero-Shot Voice Style Transfer with Only Autoencoder Loss](http://proceedings.mlr.press/v97/qian19c.html). In this repository, we demonstrate the AutoVC framework's ability to disentangle singing attributes - namely singing techniques and singing content. There are multiple aspects to consider when doing this that we hope to highlight in the forthcoming publication

```
@InProceedings{pmlr-v97-qian19c, title = {{A}uto{VC}: Zero-Shot Voice Style Transfer with Only Autoencoder Loss}, author = {Qian, Kaizhi and Zhang, Yang and Chang, Shiyu and Yang, Xuesong and Hasegawa-Johnson, Mark}, pages = {5210--5219}, year = {2019}, editor = {Kamalika Chaudhuri and Ruslan Salakhutdinov}, volume = {97}, series = {Proceedings of Machine Learning Research}, address = {Long Beach, California, USA}, month = {09--15 Jun}, publisher = {PMLR}, pdf = {http://proceedings.mlr.press/v97/qian19c/qian19c.pdf}, url = {http://proceedings.mlr.press/v97/qian19c.html} }
```


### Audio Demo

Audio demos will be uploaded in time.

### Dependencies

TBC

### Pre-trained models

The singing technique classifier and the wavenet vocoder must be downloaded separately.

| AUTO-STC | WaveNet Vocoder |
|----------------|----------------|
| [link](https://github.com/Trebolium/VocalTechClass) | [link](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi_6i7zi4nQ/view?usp=sharing) |


### 0.Convert Mel-Spectrograms

Run the ```make_spect.py```, adapting the paths as necessary to point towards your audio files directory


### 1.Mel-Spectrograms to waveform

Run ```main.py``` to start training a model


--FURTHER INSTRUCTIONS WILL FOLLOW UPON COMPLETING THIS REPOSITORY IN FULL--



