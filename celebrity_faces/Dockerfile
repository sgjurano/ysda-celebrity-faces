FROM python:3-onbuild

ADD . /app/
WORKDIR /app

RUN useradd -ms /bin/bash www
RUN chown -R www:www /app

USER www

CMD ["nohup", "python", "app.py", "&"]
