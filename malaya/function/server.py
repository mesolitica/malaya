import sys
import threading
import webbrowser
import socket
import itertools
import random
import os
import mimetypes
from http import server


def generate_handler(html, files=None):
    if files is None:
        files = {}

    class MyHandler(server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)

            if self.path == '/':
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode('utf-8'))
            else:
                this_dir = os.path.dirname(__file__)
                filepath = os.path.join(this_dir, 'web', self.path[1:])

                mimetype, _ = mimetypes.guess_type(filepath)
                self.send_header('Content-type', mimetype)
                self.end_headers()

                with open(filepath, 'rb') as fh:
                    content = fh.read()
                self.wfile.write(content)

        def log_message(self, format, *args):
            return

    return MyHandler


def find_open_port(ip, port, n=50):
    """
    Find an open port near the specified port
    """
    ports = itertools.chain(
        (port + i for i in range(n)), (port + random.randint(-2 * n, 2 * n))
    )

    for port in ports:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((ip, port))
        s.close()
        if result != 0:
            return port
    raise ValueError('no open ports found')


def serve(
    html,
    ip='127.0.0.1',
    port=8888,
    files=None,
    open_browser=True,
    http_server=None,
    **kwargs,
):
    """
    Start a server serving the given HTML, and (optionally) open a browser.

    Parameters
    ----------
    html : string
        HTML to serve
    ip : string (default = '127.0.0.1')
        ip address at which the HTML will be served.
    port : int (default = 8888)
        the port at which to serve the HTML
    files : dictionary (optional)
        dictionary of extra content to serve
    open_browser : bool (optional)
        if True (default), then open a web browser to the given HTML
    http_server : class (optional)
        optionally specify an HTTPServer class to use for showing the
        figure. The default is Python's basic HTTPServer.
    """

    Handler = generate_handler(html, files)

    if http_server is None:
        srvr = server.HTTPServer((ip, port), Handler)
    else:
        srvr = http_server((ip, port), Handler)

    if open_browser:
        def b(): return webbrowser.open(f'http://{ip}:{port}')
        threading.Thread(target=b).start()

    try:
        srvr.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        print('\nstopping Server...')

    srvr.server_close()
