import asyncio
from datetime import datetime
#Corutine
async def hello():
    print("hello")
    await asyncio.sleep(1)
    print("world")

async def say_something(what, delay):
    await asyncio.sleep(delay)
    print(what)

#Task
async def taks():

    task1 = asyncio.create_task(say_something("hello",1))
    task2 = asyncio.create_task(say_something("world",2))

    await task1
    await task2


async def task_group():

    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(say_something("Cika Vasa",1))
        task2 = tg.create_task(say_something("Potreban mi je",0.2))


async def countdown(thread, count):
    for item in range(count):
        await asyncio.sleep(1)
        print(f"Current thread: {thread} is on count : {item}")

async def geather_threads():

    result = await asyncio.gather(
        countdown("Thread1",4),
        countdown("Thread2",2)
    )
    print(f"Result is :{result}")


async def ttimeout(time):
    await asyncio.sleep(time)
    print(f"I was waiting for : {time}")


async def timer():

    try:
        await asyncio.wait_for(ttimeout(4),3)
    except asyncio.TimeoutError:
        print("TIMEOUT")

    try:
        await asyncio.wait_for(asyncio.shield(ttimeout(4)),3)
    except asyncio.TimeoutError:
        print("TIMEOUT")


async def waiting_room():

    done,pending = await asyncio.wait([ttimeout(1),ttimeout(2)])
    print(done)
    print(pending)


async def subp(command):
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()

    print(f'[{command!r} exited with {proc.returncode}]')
    if stdout:
        print(f'[stdout]\n{stdout.decode()}')
    if stderr:
        print(f'[stderr]\n{stderr.decode()}')


async def worker(queue : asyncio.Queue, name):
    while not queue.empty():
        sleep_delay,tada =await queue.get()
        await asyncio.sleep(sleep_delay)
        print(f"I was sleeping for : {sleep_delay} and {tada} , and  i am worker {name}")
        queue.task_done()
    print(f"I finished job :D {name}")

async def main_function():
    queue = asyncio.Queue()

    for item in range(7):
        queue.put_nowait((item,f'Tada :{item}'))

    tasks = []
    for item in range(3):
        task = asyncio.create_task(worker(queue,f"Worker : {item}"))
        tasks.append(task)

    #blokirajuca metoda dok se sve ne odradi
    await queue.join()
    #on cancle sve taskove sada ali ne znam sto bas
    print("Queue is empty now :D")


async def try_one():
    await  asyncio.sleep(1)
    raise asyncio.CancelledError("Muuu")

async def try_two():
    print(f"I am starting to sleep {datetime.now()}")
    await asyncio.sleep(4)
    print(f"I am finished with sleeping {datetime.now()}")
    return "Muuu"

async def cancle_me_first():
    print(f"I am starting to sleep cancle me first {datetime.now()}")
    await asyncio.sleep(6)
    print(f"I am finised with sleep cancle me first {datetime.now()}")
    print("Cigance")

async def geather_datas():

    task1 = asyncio.create_task(try_one())
    task2 = asyncio.create_task(try_two())
    # result = None
    # try:
    #     result = await asyncio.gather(try_one(),try_two())
    # except asyncio.CancelledError as e:
    #     await asyncio.sleep(0.1)
    #     print(f"Result is {result}")

    task1.cancel()
    try:
        res = await asyncio.gather(task1,task2)
        print(res)
    except asyncio.CancelledError:
        print(f"Error will start to sleep 10 sec at : {datetime.now()}")
        await asyncio.sleep(10)
        print(f"We are finished with sleeping at {datetime.now()}")


async def code_run():
    task1 = asyncio.create_task(cancle_me_first())
    print('ahahahaha')
    task2 = asyncio.create_task(try_two())
    print("muuuu")
    print("cekamo taks 1")
    await task2
    print("cekamo taks 2")
    print(task1.done())


async def check_cancel():
    taks1 = asyncio.create_task(cancle_me_first())
    taks1.cancel()
    await asyncio.shield(taks1)
    print("I am shielded")


if __name__ == "__main__":
    #asyncio.run(hello())
    #asyncio.run(taks())
    #asyncio.run(task_group())
    #asyncio.run(geather_threads())
    #asyncio.run(timer())
    #asyncio.run(waiting_room())
    #asyncio.run(subp("echo 'Petar'"))
    #asyncio.run(main_function())
    #asyncio.run(geather_datas())
    asyncio.run(code_run())
    #asyncio.run(check_cancel())