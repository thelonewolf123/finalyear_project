import socketio
import os

# from django.http import HttpResponse
# from django.contrib.sessions.models import Session
# from django.contrib.auth.models import User

basedir = os.path.dirname(os.path.realpath(__file__))
sio = socketio.Server(async_mode='eventlet')


@sio.event
def connect(sid, data):
    print(f'Socket id : {sid}')
    print('Connected')


@sio.event
def disconnect(sid):
    try:
        from .models import Online
        Online.objects.get(socketid=str(sid)).delete()
    except Exception as e:
        print(e)
    print('Disconnected')


# @sio.event
# def send_msg(sid, data):
#     try:
#         print('Message event triggered')
#         session_key = data['ssid']
#         username = data['user']
#         to_addr = data['to']
#         message = data['message']
#         session = Session.objects.get(session_key=session_key)
#         session_data = session.get_decoded()
#         print(session_data)
#         uid = session_data.get('_auth_user_id')
#         user = User.objects.get(id=uid)
#         print(user.username)

#         if(username == user.username):

#             from chat.models import Message
#             messageModel = Message.objects.create(
#                 msg_from=username, msg_to=to_addr, message=message)

#             print('Message is created')

#     except Exception as e:
#         print(e)
