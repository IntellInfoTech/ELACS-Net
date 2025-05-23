# ELACS-Net
This repository is the `PyTorch` code for our ELACS-Net.  
## 1. Introduction ##
**1) Datasets**  

Training set: [`BSDS500`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)

**2）Project structure**
```
(ELACS-Net)
|-models
|    |-__init__.py  
|    |-networks.py  
|-trained_models  
|    |-1  
|    |-4  
|    |-... (Sampling rates)
|-config.py  
|-loader.py  
|-test.py  
|-train.py
```

**3) Competing methods**

We provide a comprehensive comparison between ELACS-Net and other DL-based CS methods.
The pure model-based CS methods include MAC-Net, DPA-Net, NL-CSNet, BCS-Net, CSformer, TCS-Net, AutoBCS and MTC-CSNet, and algorithm-based unfolding methods include TransCS, DGU-Net$^+$, SODAS-Net, DPC-DUN, OCTUF, LTwIST, UFC-Net and MDGF-Net.

**4) Performance demonstrates**

![图片](https://github.com/user-attachments/assets/8224e1b8-375a-4cec-b467-9b8812a60068)
![图片](https://github.com/user-attachments/assets/8d3d74bd-e144-4442-baf6-b50d52768d01)



## 2. Usage ##

**For train:**
```
python train.py --rate=0.10 --batch_size=32
```
**For test:**
```
python test.py --rate=0.10
```
## End ##
