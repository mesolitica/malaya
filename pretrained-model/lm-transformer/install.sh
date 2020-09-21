ctpu up --zone=europe-west4-a \
--vm-only \
--disk-size-gb=300 \
--machine-type=n1-standard-8 \
--tf-version=1.15.3 \
--name=transformer-tutorial

sudo pip3 install sentencepiece
gsutil cp gs://mesolitica-tpu-general/t5-data/sp10m.cased.t5.model .