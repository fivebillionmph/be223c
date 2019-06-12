FROM python:3

ADD src /
ADD data /

# RUN only runs when image is first build
RUN pip install Flask
RUN pip install keras
RUN pip install tensorflow

#CMD runs everytimme docker starts up
