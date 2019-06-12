FROM python:3

# RUN only runs when image is first build
# CMD runs everytimme docker starts up

RUN mkdir project

ADD src /project
ADD data /project
ADD web /project
ADD scripts /project
COPY requirements.txt .

RUN apt update && apt install -y python3 python3-pip
RUN pip3 install -r requirements.txt

CMD ["/project/scripts/run-server.py"]
