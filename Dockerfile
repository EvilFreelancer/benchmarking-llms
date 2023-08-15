FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS compile-image
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
WORKDIR /app

# Install required packages
RUN set -xe \
 && apt-get -y update \
 && apt-get install -y software-properties-common curl build-essential git libaio-dev llvm-11 clang wget \
 && apt-get -y update \
 && add-apt-repository universe \
 && apt-get -y update \
 && apt-get -y install python3 python3-pip \
 && apt-get clean

# Install Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y

# Install apex
RUN set -xe \
 && pip install --upgrade pip \
 && pip install packaging==23.0 torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN set -xe  \
 && git clone https://github.com/NVIDIA/apex.git \
 && cd ./apex \
 && git checkout 8b7a1ff183741dd8f9b87e7bafd04cfde99cea28 \
 && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /app

# Install python packages
COPY requirements-xl.txt ./
RUN set -xe \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements-xl.txt

# Copy project files
COPY . .

# Install ru-gpts
RUN set -xe  \
 && echo \
 && git clone https://github.com/EvilFreelancer/ru-gpts.git ru_gpts

CMD ["sleep", "inf"]
