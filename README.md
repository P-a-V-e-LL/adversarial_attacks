# adversarial_attacks

Набор данных: ImageNet  
Архитектура нейронной сети: ResNet50

## Installation
1.Клонировать репозиторий:
```
git@github.com:P-a-V-e-LL/adversarial_attacks.git
```
2.Создать новое вируальное окружение:
```
python3.7 -m venv new_venv
source new_venv/bin/activate
```
3.Установить зависимости:
```
pip install -r requirements.txt
```
## Методы защиты

1. [Fast gradient sign method](https://arxiv.org/abs/1412.6572)
2. [Random Input Transformation](https://arxiv.org/abs/1711.01991)
3. [Pruning](https://arxiv.org/abs/1803.01442)

Для каждого метода защиты реализован скрипт обучения:
- Fast gradient sign method - resnet50_fgsm_imagenet_train.py
- Random Input Transformation - resnet50_RandomInputTransformation_imagenet_train.py
- Pruning - resnet50_pruning_imagenet_train.py

Во всех скриптах используется оптимизатор Adam.

Агрументы общие для всех скриптов:
- --data_path - путь к набору данных, по умолчанию './data/'
- --epochs - количество эпох обучения, по умочланию 250
- --batch_size - размер батча, по умолчанию 64
- --learning_rate - шаг обучения, по умолчанию 1e-3

После обучения результат будет сохранен в формате pth в папке models, также в папке loss_plots будут сохранены два графика падения ошибки: только train и train&test вместе.
