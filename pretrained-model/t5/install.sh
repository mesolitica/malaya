# make sure use ubuntu 18.04
gcloud beta compute --project=mesolitica-tpu instances create instance-1 --zone=asia-southeast1-b --machine-type=e2-standard-2 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=362309948779-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=ubuntu-1804-bionic-v20210825 --image-project=ubuntu-os-cloud --boot-disk-size=50GB --boot-disk-type=pd-standard --boot-disk-device-name=instance-1 --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
sudo apt update
sudo apt install python3-pip -y
sudo pip3 install pip -U
sudo pip3 install notebook
sudo pip3 install setuptools -U
sudo pip3 install --ignore-installed tensorflow==1.15 tensorflow-addons==0.11.1 tensorflow-datasets==1.3.2 tensorflow-estimator==1.15.1
sudo pip3 install --ignore-installed tensorflow-gan==2.0.0 tensorflow-hub==0.7.0 tensorflow-metadata==0.21.1
sudo pip3 install --ignore-installed tensorflow-probability==0.7.0 tensorflow-text==1.15.0 t5==0.5.0 tf-sentencepiece==0.1.86
sudo pip3 install --ignore-installed google-api-python-client oauth2client
sudo pip3 install --ignore-installed mesh-tensorflow==0.1.13
sudo pip3 install --ignore-installed tfds-nightly==3.1.0.dev202005080105 tensorflow-estimator==1.15.1
screen -d -m jupyter notebook --NotebookApp.token=''
# make sure tpu tensorflow version 1.15.3