import asyncio
import logging
from data_schemes._message_schemas import InformationSchema, ErrorSchema

class MessageBus:

    def __init__(self):
        self.information_queue = asyncio.Queue(maxsize=3)
        self.error_queue = asyncio.Queue(maxsize=5)

    # neblokirajuca
    async def send_message(self, message:str):
        if not message:
            error_message = ErrorSchema().load({
                "message": "Information message does not contain message "
            })
            await self.send_error_message(error_message)
        else:
            await self.information_queue.put(message)
            print(f"Message: {message} put into queue")

    async def get_message(self) -> str:
        if self.information_queue.empty():
            print("Information queue is empty")
        if self.information_queue.full():
            print("Information queue is full")
        message = await self.information_queue.get()
        # self.information_queue.task_done()
        return message

    async def send_error_message(self, message):
        try:
            self.error_queue.put_nowait(message)
        except asyncio.QueueFull as ex:
            print("Error queue is full")

    async def get_error_message(self) -> str:
        try:
            message = self.error_queue.get_nowait()
            return message
        except asyncio.QueueEmpty as ex:
            print("Error queue is empty")


    async def run_write(self):
        counter = 0
        while True:
            await self.send_message(f"Task {counter} is running")
            await asyncio.sleep(0.5)
            counter += 1

    async def run_read(self):
        while True:
            await self.get_message()
            await asyncio.sleep(1)

async def run():
    m_b = MessageBus()
    await m_b.run_write()
    await m_b.run_read()





asyncio.run(run())