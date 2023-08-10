# PuppetMaster
Deep Learning-Based Approach for Creation of High Resolution Synthetic Videos
## Introduction
This model brings forward a new approach to the image animation model for the generation of a high-quality synthetic video sequence without any initial information or annotations. The proposed multimodal architecture makes use of two models for image animation and video enhancement. The model once trained on a set of videos belonging to the same category (e.g. human bodies, faces) can generate high- resolution video sequences of any object belonging to the same class. The model outperforms SOTA first-order motion models for unsupervised video retargeting.

## Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fC8gjagj1dyKT9vtICqv-06yWkP5I8P7?usp=sharing)

### Guidelines
1. Select GPU runtime
3. Execute all cells
4. After executing the last cell
   1. Execute The Cell
   2. Open Link
   3. Copy External URL without port number
   4. Paste URL in textbox of URL opened


## Architecture
### PuppetMaster Architecture
![model drawio](https://github.com/githubhamza/PuppetMaster/assets/98768916/ac2b4ae9-cac2-4ff8-9a04-b5ea3d696f81)

### Residual in Residual Dense Block (RRDB) Architecture
![RRDB drawio](https://github.com/githubhamza/PuppetMaster/assets/98768916/6dea8e98-42d1-49ca-8aff-b43c70d1afd4)


## Sample
### Source Image

![1(S)](https://github.com/githubhamza/PuppetMaster/assets/98768916/e478de54-b4bf-4ae0-814a-d6c9b20966d0)

### Driving Video


https://github.com/githubhamza/PuppetMaster/assets/98768916/8614dba3-0566-422d-ad82-68900ded4c21

### First Order Motion Model Result



https://github.com/githubhamza/PuppetMaster/assets/98768916/dbef62ae-025e-4140-aab2-ba4c47aeb737

### Proposed Model Result


https://github.com/githubhamza/PuppetMaster/assets/98768916/7455a567-b627-41f3-b772-366fcdeb63b8

## Evaluation
The evaluation is done using the following three matrices:
1. **Mean Squared Error (MSE):** Lower is better. 
2. **Mean Absolute Error (MAE):** Lower is better. 
3. **PSNR (Peak Signal-to-Noise Ratio):** Higher is better. 


| **Model** | **MSE** | **MAE** | **PSNR** |
|---|---|---|---|
| **First Order Motion Model** | 71.15 | 143.48 | 29.69 |
| **Proposed** | 69.17 | 138.00 | 29.82 |

## Custom Setup
1. Use Python Version 3.7
2. Install libraries using `pip install requirements.txt`
3. Download the config folder from [config](https://1drv.ms/f/s!Amu2_EOykJZIgQzDm2yZGSv8kinw?e=phTu0G)
4. Place the config folder inside the `PuppetMaster` directory
5. Clone the repository `!git clone https://github.com/xinntao/Real-ESRGAN.git`
6. Navigate inside the cloned directory and run `!python setup.py develop`

## Inspired By
Special thanks to all contributors of the following repositories:
1. [First-Order-Model](https://github.com/AliaksandrSiarohin/first-order-model)
2. [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

