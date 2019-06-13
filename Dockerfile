FROM python:3

# RUN only runs when image is first build
# CMD runs everytimme docker starts up

RUN mkdir project

COPY src /project/src
COPY data /project/data
COPY web /project/web
COPY scripts /project/scripts
COPY requirements.txt .

RUN pip3 install -r requirements.txt

CMD ["/project/scripts/run-server.sh"]
