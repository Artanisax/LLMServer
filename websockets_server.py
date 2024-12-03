import asyncio
from websockets import serve

async def echo(websocket):
    print("connected")
    async for message in websocket:
        print("recv:", message)
        await websocket.send(message)

async def main():
    async with serve(echo, "localhost", 8080):
        await asyncio.get_running_loop().create_future()  # run forever

if __name__ == "__main__":
    print("start serving")
    asyncio.run(main())