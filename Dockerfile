FROM python:3

ADD src /
ADD data /

# RUN only runs when image is first build
RUN pip install Flask

#CMD runs everytimme docker starts up
