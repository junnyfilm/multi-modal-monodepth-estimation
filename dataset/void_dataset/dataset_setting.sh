apt-get clean && apt-get update 
apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    curl \
    lsb-release

#### ros install #####
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
####### ubuntu 16.04 ##########
# apt-get update 
# apt-get install ros-kinetic-desktop-full
# apt install -y python-rosinstall python-rosinstall-generator python-wstool build-essential
######## ubuntu 20.04 ###########
apt update 
apt install -y ros-noetic-desktop-full

echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
rosdep init
rosdep update


# bash bash/setup_dataset_void_raw.sh
# ## 다음 링크에서 google drive로 파일 다운로드
# ## https://drive.google.com/open?id=1bTM5eh9wQ4U8p2ANOGbhZqTvDOddFnlI
# mkdir void_release
# unzip -o void_1500.zip -d void_release/
# bash bash/setup_dataset_void.sh unpack-only