# Настройка

1. Cоздание виртульаной среды (virtual enviroment)

Чтобы изолировать версии всех библиотек Питона от установленных в системе
```bash
# установка
sudo apt install python3-venv

# Создание виртуальной среды
python -m venv my-venv

# Активация
source my-venv/bin/activate

# Выключение
deactivate
```


2. Установка библиотек (это несколько минут)
```bash
pip3 install -r requirements.txt
```


3. Клонирование репозитория с YoloFace
```bash
git clone https://github.com/elyha7/yoloface
```

Убрать предупреждения о ресайзе изображения
yoloface/utils/general.py строка 93



При первом запуске скачаются веса нейронки для детекции (111 Мб)