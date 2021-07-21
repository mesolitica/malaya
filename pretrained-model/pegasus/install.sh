# run in cloud shell
ctpu up --zone=europe-west4-a \
--vm-only \
--disk-size-gb=100 \
--machine-type=n1-standard-2 \
--tf-version=1.15.3 \
--name=transformer-tutorial \
--project=mesolitica-tpu
gcloud beta compute ssh --zone "europe-west4-a" "ubuntu"@"transformer-tutorial"  --project "mesolitica-tpu"
git clone https://github.com/huseinzol05/malaya.git
cd malaya/pretrained-model/pegasus