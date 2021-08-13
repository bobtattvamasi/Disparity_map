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


#------

# Обновленный документ

1) Запуск  
  Тестовый режим:
  - Для тестового режима надо скачать папки [изображений](https://drive.google.com/drive/folders/1s07Tic0D12pmU0DAij_0LdIbgPd_onRS?usp=sharing) и [фонового изображения](https://drive.google.com/drive/folders/1DJMLue_h7pLnrZPJnNma6e6_aQVwIOLL?usp=sharing) и положить в папку data/  
  - Так же в папке  data/ должны лежать калибровочные данные в папке calibration_data/ И внутри этой папки должна лежать папка с названием разрешения одной из камер . По умолчанию стоит значение 720. Слудовательно должна быть папка под названием 720p. Внутри этой папки должны быть 3 .npz файла. [Ссылка](https://drive.google.com/drive/folders/1dWrms-0M5oWDNn8TZBEo2vvf0rvj2g4q?usp=sharing) на такую папку 720p. Как их получитт будет ниже.
  - И затем запустить приложение: python3 main.py -m 0
  - Вы увидите следующее окно:  
  
  - 
3) Калибровка 
