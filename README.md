# ysda-celebrity-faces


# Общее описание системы
Сервис построен на основе двух docker-контейнеров, связываемых через http, так же написан рецепт для docker-compose.
В контейнере celebrity_faces находится веб-мордочка, скрипт для поиска лиц на залитой фотографии на основе предобученных каскадов Хаара (opencv) и нейронная сеть facenet, которая для каждого найденного лица считает 128-мерный эмбеддинг.
Во втором контейнере hnsw_index находится написанный на C++ HNSW индексер, метод поиска kNN которого проброшен в flask через cython-биндинг.

# Текущее состояние
В веб-форму можно залить произвольную фотографию из которой будут вырезаны распознанные лица, для каждого из которых будут найдены ближайшие соседи.

В выдаче показывается 5 ближайших соседей а также лицо, восстановленное из переданных эмбеддингов картинки и ближайшего соседа, для оригинальной фотографии, масштабированной до 160х160, необработанного opencv-кропа и расширенного opencv-кропа с эмпирически подобранными коэффициентами.

Результаты работы и более подробное описание принятых решений можно увидеть в отчёте. Результаты экспериментов с генерацией лиц лежат в репозитории, папка `experiments`


# Использованные статьи и ресурсы
* Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs https://arxiv.org/abs/1603.09320
* Face Recognition using Tensorflow https://github.com/davidsandberg/facenet
* Face Detection using Haar Cascades https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
* Полное практическое руководство по Docker: с нуля до кластера на AWS https://habrahabr.ru/post/310460/

# Запуск
Клонируем репозиторий
```
git clone git@github.com:sgjurano/ysda-celebrity-faces.git
```

Скачиваем параметры для индексера, модель для подсчета эмбеддингов, GAN, и датасет отсюда (датасет надо распаковать): https://yadi.sk/d/NkJaLhTS3UHBWd
```
unzip lsml-celebrity-faces -d .
```

Теперь нужно собрать докер-контейнеры для обеих частей:
```
cd hnsw_index && ./run_docker.sh
cd celebrity_faces && ./run_docker.sh
```

Запустим оба контейнера, смонтировав ресурсы и прокинув соответствующие порты:
```
docker-compose up
```

