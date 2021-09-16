
sudo apt update
sudo apt install python3-pip pkg-config -y
sudo apt install python3-venv python3-wheel -y
sudo pip3 install pip -U
pip3 install virtualenv
git clone https://github.com/zihangdai/xlnet
cd xlnet
python3 -m venv env
source env/bin/activate
pip3 install wheel
pip3 install tensorflow==1.13.1 sentencepiece==0.1.91
pip3 install google-api-python-client oauth2client