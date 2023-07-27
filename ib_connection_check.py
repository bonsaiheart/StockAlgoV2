import asyncio
import subprocess

async def check_ib_connection():
    while True:
        try:
            process = await asyncio.create_subprocess_shell(
                'python main.py check_ib_connection',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            output = stdout.decode().strip() + stderr.decode().strip()
            print(f"IB Connection Status: {output}")
        except asyncio.CancelledError:
            break
        await asyncio.sleep(300)  # Wait for 5 minutes before checking again

async def main():
    await check_ib_connection()

if __name__ == "__main__":
    asyncio.run(main())
