NUM_WORKER=$1
BIND_ADDR=0.0.0.0:8080
python /app/load_model.py
gunicorn --graceful-timeout 30 --timeout 180 -w $NUM_WORKER -b $BIND_ADDR -k sync app
