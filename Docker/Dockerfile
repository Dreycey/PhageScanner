# Use Ubuntu as the base image
FROM ubuntu:22.04

# Avoid prompts from apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Install necessary tools
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip

# Install BLAST
RUN wget https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/ncbi-blast-2.15.0+-x64-linux.tar.gz && \
    tar -xvf ncbi-blast-2.15.0+-x64-linux.tar.gz && \
    cp ncbi-blast-2.15.0+/bin/* /usr/local/bin/

# Install CD-HIT
RUN wget https://github.com/weizhongli/cdhit/releases/download/V4.8.1/cd-hit-v4.8.1-2019-0228.tar.gz && \
    tar -zxvf cd-hit-v4.8.1-2019-0228.tar.gz && \
    cd cd-hit-v4.8.1-2019-0228/ && \
    make && \
    cp cd-hit /usr/local/bin

# Install MEGAHIT
RUN wget https://github.com/voutcn/megahit/releases/download/v1.2.9/MEGAHIT-1.2.9-Linux-x86_64-static.tar.gz && \
    tar -zxvf MEGAHIT-1.2.9-Linux-x86_64-static.tar.gz && \
    cp MEGAHIT-1.2.9-Linux-x86_64-static/bin/megahit* /usr/local/bin/

# Install PHANOTATE
RUN wget https://github.com/deprekate/PHANOTATE/archive/refs/tags/1.5.0.tar.gz && \
    tar -zxvf 1.5.0.tar.gz && \
    cd PHANOTATE-1.5.0/ && \
    python3 setup.py install

# Clone and install PhageScanner
RUN git clone https://github.com/Dreycey/PhageScanner.git && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    cd PhageScanner && \
    git checkout dreycey/updateInstall && \
    python3 -m pip install -r requirements.txt

# Copy and set the entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the working directory
WORKDIR /data

# Set the entrypoint and default command
ENTRYPOINT ["entrypoint.sh"]
CMD ["--help"]