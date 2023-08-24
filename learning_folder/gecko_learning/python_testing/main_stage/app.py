from data_schemes._message_schemas import InformationSchema
from pyt_async._message_queue import MessageBus

import asyncio


class Tasks:

    def __init__(self, reading_workers, writting_workers):
        self.read = reading_workers
        self.write = writting_workers
        self.message_bus = MessageBus()

    async def taks_writte_to_bus(self, message):
        mes = InformationSchema().load({"message": message})
        await self.message_bus.send_message(mes)



    async def write_all_tasks(self):
        for item in range(self.write):
            await self.taks_writte_to_bus(f"Task {item} writes to queue")
            await asyncio.sleep(1)




tasks = Tasks(3,5)
asyncio.run(tasks.write_all_tasks())