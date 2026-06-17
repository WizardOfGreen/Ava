FROM gcc:15 AS final
WORKDIR /app 
COPY . . 

RUN apt-get update 
RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \ 
    && rm -rf /var/lib/apt/lists/*

RUN cmake --version 
RUN rm -r .\build
RUN cmake -S . -B build
RUN cd.\build\