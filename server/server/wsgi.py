"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.1/howto/deployment/wsgi/
"""

import eventlet.wsgi
import eventlet
from socket_app.views import sio
from socketio import Middleware
import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")

django_app = get_wsgi_application()
application = Middleware(sio, django_app)

eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 8000)), application)
