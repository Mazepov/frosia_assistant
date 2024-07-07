#Импорт необходимых бибилиотек
import speech_recognition as sr
import pyttsx3
import pyaudio
import sounddevice as sd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pyautogui
import requests
import cv2
import time
import transliterate #Из кириллицы в латиницу
import tensorflow as tf
import numpy as np


#Смотрим на наличие необходимой аппаратуры (удалить)
# print(sd.query_devices())

# #Проверка работы камеры
# ret, img = cv2.VideoCapture(0).read()
# try:
#     img.any() == None
# except:
#     print('Я вас не вижу! Проверьте работу камеры')


# Функция голоса Фроси
def Frosia_speak(Frosia_audio):
    # Инициализируем Фросю
    start = pyttsx3.init()
    start.say(Frosia_audio)
    start.runAndWait()

# Функция команды пользователя
def listen_command():
    # Инициализируем распознование речи
    r = sr.Recognizer()

    #Пауза, после которой начнётся запись голоса
    r.pause_threshold = 0.5
    print('Говорите')
    Frosia_speak('Пук!')
    #Запускаем микрофон
    with sr.Microphone() as mic:
        #Лишний шум источник микрофон,
        r.adjust_for_ambient_noise(source=mic, duration=0.5)
        try:
            #Запрос голосом с микрофона, timeout - время пока Фрося ждет отклика,
            audio = r.listen(source=mic, timeout=3)
            #Распознаем речь в текст с помощью google (требуется интернет)
            user_command = r.recognize_google(audio_data=audio, language='ru-RU').lower()
        except:
            message = 'Не поняла вас. Повторите, пожалуйста'
            Frosia_speak(message)
            user_command = listen_command()
    return user_command
    # Frosia_speak(f'Вы сказали - {user_command}')


### ЧАСТЬ 1 ###

#Список пользователей
users_directory = "C:/Users/admin/Desktop/Projects/Assistent_Frosia/pythonProject1/Photo"
# Получаем список файлов
users_list = os.listdir(users_directory)
#Создадим папку с фотографией человека перед экраном
os.makedirs('C:/Users/admin/Desktop/Projects/Assistent_Frosia/pythonProject1/Current_user_temp_photo', exist_ok=True)


# иницилизируем детектор с помощью каскадов Хаара
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



#Функция сделать фото человека перед экраном
def current_user_unknown_photo():
    cap = cv2.VideoCapture(0)
    # "Прогреваем" камеру, чтобы снимок не был тёмным
    for i in range(5):
        cap.read()
    ret, frame = cap.read()
    cv2.imwrite('C:/Users/admin/Desktop/Projects/Assistent_Frosia/pythonProject1/Current_user_temp_photo/Current_user.png', frame)


#Имя новое функция
def new_user_name(users_list):
    Frosia_speak('Назовите ваше имя или фамилию')
    user_name = listen_command()
    if user_name in users_list:
        Frosia_speak('Пользователь с таким именем уже существует!')
        new_user_name(users_list)
    else:
        return user_name

# Делаем фото функция
def make_photo(u_path):
    cap = cv2.VideoCapture(0)
    # "Прогреваем" камеру, чтобы снимок не был тёмным
    for i in range(5):
        cap.read()
    Frosia_speak('Пожалуйста, смотрите ровно в камеру 5 секунда, слегка наклоняя голову')
    for i in range(5):
        ret, frame = cap.read()
        cv2.imwrite(u_path + f'/cam_{i}.png', frame)
        time.sleep(1)
    Frosia_speak('Поворачивайте голову налево в течении 5 секунд, слегка наклоняя голову')
    for i in range(5,10):
        ret, frame = cap.read()
        cv2.imwrite(u_path + f'/cam_{i}.png', frame)
        time.sleep(1)
    Frosia_speak('Поворачивайте голову направо в течении 5 секунд, слегка наклоняя голову')
    for i in range(10,15):
        ret, frame = cap.read()
        cv2.imwrite(u_path + f'/cam_{i}.png', frame)
        time.sleep(1)
    Frosia_speak('Поднимайте голову наверх в течении 5 секунд, слегка наклоняя голову')
    for i in range(15,20):
        ret, frame = cap.read()
        cv2.imwrite(u_path + f'/cam_{i}.png', frame)
        time.sleep(1)
    Frosia_speak('Опускайте голову вниз в течении 5 секунд, слегка наклоняя голову')
    for i in range(20,25):
        ret, frame = cap.read()
        cv2.imwrite(u_path + f'/cam_{i}.png', frame)
        time.sleep(1)
    # Отключаем камеру
    cap.release()

#Подготовка для инференса модели
def model_in_user(path_curr_user):
    img = np.empty((1, 224, 224, 3))  # Заготовка под картинку
    # Чтение картинки (Необходимо взять любую картинку с лицом для теста инференса)
    img_c_u = cv2.imread(path_curr_user)
    # Обработка
    img_test_final = tf.image.resize(img_c_u, (224, 224))
    # Конвертация в нужный формат
    img[0] = img_test_final
    return img

### ЧАСТЬ 2 ###

# Функция сворачивания всех окон
def wrp_all_windows():
    pyautogui.hotkey('winleft', 'd')

#Функция погоды
def weather():
    Frosia_speak('Назовите город')
    city = listen_command()
    url = 'https://api.openweathermap.org/data/2.5/weather?q=' + city + '&units=metric&lang=ru&appid=79d1ca96933b0328e1c7e3e7a26cb347'
    weather_data = requests.get(url).json()
    temperature = round(weather_data['main']['temp'])
    temperature_feels = round(weather_data['main']['feels_like'])
    Frosia_speak(f'Сейчас в городе {city} температура {temperature}, ощущается как {temperature_feels}')

#Основное действие Фроси
def Frosia_actions(comm):
    if comm == 'погода':
        weather()
        Frosia_speak('Что нибудь ещё?')
        comm = listen_command()
        Frosia_actions(comm)

    elif comm == 'свернуть':
        wrp_all_windows()
        Frosia_speak('Что нибудь ещё?')
        comm = listen_command()
        Frosia_actions(comm)

    else:
        Frosia_speak('До свидания, кожанный ублюдок!')




#СТАРТ

Frosia_speak('  Приветствую, Меня зовут Фрося! Все команды говорите после звукового сигнала Пук!')



#Делаем фото того кто сидит перед экраном
current_user_unknown_photo()


################################################################################################################################################################################
#СОЗДАНИЕ ПОЛЬЗОВАТЕЛЯ / НАЧАЛО РАБОТЫ
#ОТРАБОТКА НС. Если с вероятностью Фрося не знает пользователя то запускается знакомство

def get_to_know_frosia():
    # ЗНАКОМСТВО
    Frosia_speak('Давайте познакомимся!')

    # Создаем папку с фотографиями пользователя
    current_user = new_user_name(users_list)

    # Создаем папку латиницей
    current_user_eng = transliterate.translit(current_user, reversed=True)

    # Словарь имен кириллица-латиница
    users_transl_dict = {current_user: current_user_eng}

    os.mkdir(users_directory + '/' + current_user_eng)
    Frosia_speak(f'Отлично, {current_user}! Приятно познакомится!')
    Frosia_speak('Теперь мне необходимо сделать ваши фотографии, чтобы я смогла вас запомнить и узнать в следующий раз!')

    # Делаем фото
    current_user_path = users_directory + '/' + current_user_eng

    make_photo(current_user_path)
    Frosia_speak(f'Спасибо, {current_user}! Мы успешно сделали ваши фотографии. Теперь мне нужно немного времени, чтобы обучить мою нейронную сеть вас узнавать.')

    # Нейронная сеть
    # Запуск JupiterNotebook с нейронной сетью
    Frosia_speak(f'Пока нейронная сеть обучается вы можете подумать над загадкой: . Я сообщу вам когда процесс обучения будет завершен.')

    Frosia_speak(f'Процесс обучения завершен!')



#Проверяем пустая ли папка с пользователями (в папке по умолчанию находится базовый пользователь)
if len(os.listdir(users_directory)) == 1:
    get_to_know_frosia()

#Загружаем модель
n_faces = len(os.listdir(users_directory))
model_path = f'C:/Users/admin/Desktop/Projects/Assistent_Frosia/pythonProject1/NN_model/nn_model_recognition_user/checkpoint_best_user_count_{n_faces}.h5'
model_user_recognation = tf.keras.models.load_model(model_path)

def recognition_user():
    #Подготовка изображения текущего пользователя
    curr_user_img = model_in_user('C:/Users/admin/Desktop/Projects/Assistent_Frosia/pythonProject1/Current_user_temp_photo/Current_user.png')

    #Инференс модели (predict)
    pred_user = model_user_recognation.predict(curr_user_img)

    #Кто есть кто:
    users_dict = {1: "sergej", 0: "dasha"}

    #Определяем имя и вероятность
    name = users_dict[np.argmax(pred_user[0])]
    probability = round(pred_user[0].max(), 3)
    print(name, probability)
    return name, probability

name, probability = recognition_user()

def final_recognition_frosia_action(name, probability):
    if probability > 0.80:
        #Итоговый результат работы классификатора
        Frosia_speak(f'С веротностью {probability} вас зовут {name}! Это так?')
        comm_user_name = listen_command()
        if comm_user_name == 'да':
            Frosia_speak(f'Здравствуйте {name}! Чтобы вы хотели?')
            comm = listen_command()
            Frosia_actions(comm)
        elif comm_user_name == 'нет':
            Frosia_speak('Попробуем ещё раз!')
            current_user_unknown_photo()
            name, probability = recognition_user()
            final_recognition_frosia_action(name, probability)
        else:
            Frosia_speak('Я вас не знаю, так что до свидания, кожанный ублюдок!')
    else:
        Frosia_speak('Что то я вас не узнаю! Мы знакомы?')
        comm_user_name = listen_command()
        if comm_user_name == 'да':
            Frosia_speak('Попробуем ещё раз!')
            current_user_unknown_photo()
            name, probability = recognition_user()
            final_recognition_frosia_action(name, probability)
        elif comm_user_name == 'нет':
            get_to_know_frosia()

final_recognition_frosia_action(name, probability)

################################################################################################################################################################################


# #Тестирование
# comm = listen_command()
# print(comm)
# Frosia_actions(comm)


# def main():
#     user_command = listen_command()
#
# if __name__== '__main__':
#     main()