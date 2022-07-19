=jsk py ```py
import time
import asyncio
async def handle_server(server, timestamp):
    server = await _bot.fetch_guild(server,with_counts=True)
    async with _bot.db.conn.cursor() as cur:
        s = await cur.execute('select timestamp from serveractivity where id = ?',server.id)
        return server.id, ((time.time() - (await s.fetchall())[0][0])/2419200)/(server.approximate_member_count/70)
async with _bot.db.conn.cursor() as cur:
    s = await cur.execute('select * from serveractivity')
    servers = [row[:] for row in await s.fetchall()]
    async def handle_servers():
        return asyncio.gather(*(handle_server(server, timestamp) for server, timestamp in servers))
    print(await handle_servers())
```