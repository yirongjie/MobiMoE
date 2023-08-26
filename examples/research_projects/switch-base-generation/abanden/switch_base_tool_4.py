
# import time
# import asyncio
# import pypeln as pl

# globb = 1


# from concurrent.futures import ThreadPoolExecutor
# _executor = ThreadPoolExecutor(20)


# def printfunc(x):
#     for i in range(10):
#         global globb
#         globb  = globb+1
#         print(x, i, globb)

# async def load_and_inference_model(x):
    
#     loop = asyncio.get_event_loop()
#     await loop.run_in_executor(_executor, printfunc, str(x))


#     return 0

# async def main():

#     data = range(5) 

#     stage = pl.task.map(load_and_inference_model, data, workers=3, maxsize=4)

#     data = await stage 


# start_time = time.perf_counter()

# asyncio.get_event_loop().run_until_complete(main())

# end_time = time.perf_counter()

# # 计算执行时间（毫秒）
# execution_time = (end_time - start_time) * 1000
# print("代码块执行时间为:", execution_time, "毫秒")


import pypeln as pl
import asyncio
from random import random
from concurrent.futures import ThreadPoolExecutor
_executor = ThreadPoolExecutor(20)
globb = 0
def printfunc(x):
    for i in range(10):
        global globb
        globb  = globb+1
        print(x, i, globb)
def printfunc2(x):
    for i in range(10):
        global globb
        globb  = globb+1
        print("-----",x, i, globb)
async def slow_add1(x):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, printfunc, str(x))
    # await asyncio.sleep(random()) # <= some slow computation
    return x + 1

async def slow_gt3(x):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, printfunc2, str(x))
    # await asyncio.sleep(random()) # <= some slow computation
    return x > 3

data = range(1) # [0, 1, 2, ..., 9] 

stage = pl.task.map(slow_add1, data, workers=3, maxsize=4)
# stage = pl.task.filter(slow_add1, data, workers=2)
stage = pl.task.filter(slow_gt3, stage, workers=2)

data = list(stage) # e.g. [5, 6, 9, 4, 8, 10, 7]
