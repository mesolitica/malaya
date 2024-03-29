import sys
import os

IMPORT_LOCAL = os.environ.get('IMPORT_LOCAL', 'false') == 'true'

if IMPORT_LOCAL:
    SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
    sys.path.insert(0, SOURCE_DIR)

from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import FastAPI
from playwright.async_api import async_playwright
from PIL import Image
import io
import playwright
import asyncio

app = FastAPI()

page = None


async def video_streamer(client_id):
    global page
    f = 'black.jpg'
    if IMPORT_LOCAL:
        f = os.path.join('./app', f)
    with open(f, 'rb') as fopen:
        black_frame = fopen.read()
    while True:
        if page is None:
            frame = black_frame
        else:
            r = await page.screenshot()
            image = Image.open(io.BytesIO(r))
            image = image.convert('RGB')
            jpeg_bytes = io.BytesIO()
            image.save(jpeg_bytes, format='JPEG')
            frame = jpeg_bytes.getvalue()

        yield (
            b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

        await asyncio.sleep(0.05)


@app.get('/initialize')
async def initialize():
    global page
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch()
    page = await browser.new_page()


@app.get('/goto')
async def goto(website: str):
    global page
    if page is None:
        await initialize()
    try:
        await page.goto(website)
        return True
    except BaseException:
        return False


@app.get('/')
async def get():
    f = 'index.html'
    if IMPORT_LOCAL:
        f = os.path.join('./app', f)
    with open(f) as fopen:
        html = fopen.read()
    return HTMLResponse(html)


@app.get('/video_feed')
async def video_feed(client_id: str):

    return StreamingResponse(
        video_streamer(client_id=client_id),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )
