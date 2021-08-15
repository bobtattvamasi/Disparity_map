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

# Инструкция  

1) Запуск  
  Тестовый режим:
  - Для тестового режима надо скачать папки [изображений](https://drive.google.com/drive/folders/1s07Tic0D12pmU0DAij_0LdIbgPd_onRS?usp=sharing) и [фонового изображения](https://drive.google.com/drive/folders/1DJMLue_h7pLnrZPJnNma6e6_aQVwIOLL?usp=sharing) и положить в папку data/  
  - Так же в папке  data/ должны лежать калибровочные данные в папке calibration_data/ И внутри этой папки должна лежать папка с названием разрешения одной из камер . По умолчанию стоит значение 720. Слудовательно должна быть папка под названием 720p. Внутри этой папки должны быть 3 .npz файла. [Ссылка](https://drive.google.com/drive/folders/1dWrms-0M5oWDNn8TZBEo2vvf0rvj2g4q?usp=sharing) на такую папку 720p. Как их получитт будет ниже.
  - И затем запустить приложение: python3 main.py -m 0 -d 0
  опция -m или --mode - означает один из двух режимов работы приложения. m = 1, означает что приложение ищет и подключается к стереокамере через PiCamera. m=0 означает что приложение будет работать в тестовом режиме. Вместо видеопотока будут картинки из папки data/TEST_IMAGES
  опция -d или --debug означает либо d=0 - карта глубин строится по нажатию на соответсвующую кнопку; d=1 карта глубин строится постояно

  - Для отображения 3д-модели нужно установить в систему приложение meshlab.  
    ```bash 
    apt install meshlab
    ```  
    С помощью него и отображается облако точек формата .ply
  - 
2) Калибровка 

Для получения результатов калибровки используется [этот проект](https://github.com/realizator/stereopi-fisheye-robot). Иструкция в нем же. Вам понадобятся 1,2 и 3 скрипты. 1ый - это тест, что у вас правильно подключенно и установлено. 2ой нужно проходить с распечатанной шахматной доской(4 черных квадрата на 5 черных квадратов). Лучше эту распечатку приклеить или пригвоздить на что-то ровное(например картонку). Дальше рекомендации по процессу калибровки: Нужно на камеру показывать шахматную доску с маленькими углами. В итоге за все 30 снимков на основании которых алгоритм будет калибровать, необходимо что бы доска побывала во всех частях камеры. 3ий скрипт берет полученные изображения и на основании их производится калибровка. Если все прошло удачно, что можно заходить в папку calibration_data и брать от туда папку и импортировать в этот проект.  Если ошибки, то возможно стоит просто начать все заново. Так же в конце третьего скрипта будут картинки если он будет выполнен. Если на них ничего не понятно или все слишком закрученно, то опять же, лучше начать заново.

3) [Инструкция](https://docs.google.com/document/d/1uryWXAtMaZfHeCOyuYQR5KJU7uKrwXc4l5hDJktKYy0/edit?usp=sharing) по использованию программой Disparity_map  
4) [Файловая структура проекта](https://docs.google.com/document/d/1Mhv_-v0SP36MGn2bDBBo-wA3IVT2hvjS4ipjb-d1uW4/edit?usp=sharing)
