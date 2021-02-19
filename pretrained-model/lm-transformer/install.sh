# run in cloud shell
ctpu up --zone=europe-west4-a \
--vm-only \
--disk-size-gb=100 \
--machine-type=n1-standard-1 \
--tf-version=1.15.3 \
--name=tpu-vm \
--project=mesolitica-tpu

sudo pip3 install sentencepiece
gsutil cp gs://mesolitica-tpu-general/t5-data/sp10m.cased.t5.model .