TORCHVISION_VERSION=0.9.0
PYTORCH_VERSION=1.8.0
python=3.8
PYTHON_VERSION=${python}

apt-get clean && apt-get update 
apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    curl \
    wget \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    ffmpeg \
    python${PYTHON_VERSION}-distutils \
    python3-testresources \
    git

apt-get install -y tmux

ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python
ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

pip install numpy Pillow matplotlib pandas
pip install -r requirements.txt
# pip install torch==${PYTORCH_VERSION}+cu111 torchvision==${TORCHVISION_VERSION}+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# ########## ROS install ######################################
# apt-get clean && apt-get update 
# apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
#     curl \
#     lsb-release


# sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# ####### ubuntu 16.04 ##########
# # apt-get update 
# # apt-get install ros-kinetic-desktop-full
# # apt install -y python-rosinstall python-rosinstall-generator python-wstool build-essential
# ######## ubuntu 20.04 ###########
# apt update 
# apt install -y ros-noetic-desktop-full

# echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
# source ~/.bashrc
# # apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
# # rosdep init
# # rosdep update
