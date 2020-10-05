# Disparity_map

![](https://miro.medium.com/max/640/1*B8XA5sXUeUSY26Kl1Y_dew.gif 'Да, эта гифка с того проекта')

It's work is based on this repository:
https://github.com/realizator/stereopi-fisheye-robot

# How to run code?
![](https://vision.middlebury.edu/stereo/data/scenes2001/data/anigif/reproj_inv/tsukuba_ri_b.gif 'Да, этот код высчитывает отклонения')

You will need python3; cameraPi as library and other stuff:

```bash
pip install -r requirements.txt

```


So if you have a stereocamera enable in your system, so just run code:

```bash
python main.py -m 1
```
options:
m: 0 - working on pictures, 1 - working with video from stereoPiCamera
d: 0 - depth map created by button, 1 - always

