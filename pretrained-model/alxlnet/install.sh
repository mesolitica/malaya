sudo apt update
sudo apt install python3-pip -y
sudo pip3 install pip -U
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
sudo pip3 install tensorflow==1.13.1 sentencepiece 
sudo pip3 install google-api-python-client oauth2client