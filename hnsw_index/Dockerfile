FROM ubuntu:17.10

RUN apt-get update -y
RUN apt-get install libstdc++6 g++ python3 python3-pip -y

ADD . /app/
WORKDIR /app
RUN /usr/bin/python3 -m pip install -r requirements.txt

WORKDIR /app/cpp
RUN /usr/bin/python3 setup.py build_ext -i
RUN mv pyhnsw.cpython-36m-x86_64-linux-gnu.so /app/

RUN useradd -ms /bin/bash www
RUN chown -R www:www .

USER www
WORKDIR /app
CMD ["nohup", "/usr/bin/python3", "app.py", "&"]
