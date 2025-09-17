import json
from channels.generic.websocket import AsyncJsonWebsocketConsumer
from .ai_utils import ask_ai
from asgiref.sync import sync_to_async

class ChatConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self):
        await self.accept()
        
        
    async def disconnect(self, close_code):
        print(f"Ai disconnect with code: {close_code}")
        
        
    async def receive(self, text_data):
        data = json.loads(text_data)
        user_message = data["message"]
        
        ai_response = await sync_to_async(ask_ai)(user_message)

        await self.send(text_data=json.dumps({
            "user_message": user_message,
            "ai_response": ai_response
        }))