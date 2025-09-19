import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .models import Message, ChatRoom
from .views import get_ai_response
from asgiref.sync import sync_to_async

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'chat_{self.room_id}'
        
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        
        room = await sync_to_async(ChatRoom.objects.get)(id=self.room_id)
        new_msg = await sync_to_async(Message.objects.create)(
            room=room,
            user_message=message
        )
        
        history = await sync_to_async(list)(Message.objects.filter(room=room).order_by('timestamp'))
        ai_response = await sync_to_async(get_ai_response)(message, history[:-1])
        
        new_msg.ai_response = ai_response
        await sync_to_async(new_msg.save)()
        
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message,
                'ai_response': ai_response
            }
        )

    async def chat_message(self, event):
        message = event['message']
        ai_response = event['ai_response']
        
        await self.send(text_data=json.dumps({
            'message': message,
            'ai_response': ai_response
        }))