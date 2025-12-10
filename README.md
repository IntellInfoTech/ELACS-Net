# ELACS-Net
This repository is the `PyTorch` code for our ELACS-Net.  
## 1. Introduction ##
**1) Datasets**  

Training set: [`BSDS400`](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)

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

<img width="1865" height="791" alt="image" src="https://github.com/user-attachments/assets/b1c52df1-45d5-40ea-84c6-b0753aeb19b6" />
<img width="1857" height="787" alt="image" src="https://github.com/user-attachments/assets/98f285d6-5162-43d9-a812-32ded3f4a32b" />


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
