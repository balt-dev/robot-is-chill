from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Coroutine, Optional
from .db import Database

import discord
from discord.ext import commands
from PIL import Image

if TYPE_CHECKING:
    from .cogs.render import Renderer
    from .cogs.variants import VariantHandlers

class Context(commands.Context):
    async def error(self, msg: str) -> discord.Message: ...
    async def send(self, content: str="", embed: Optional[discord.Embed] = None, **kwargs) -> discord.Message: ...

class Bot(commands.Bot):
    db: Database
    cogs: list[str]
    embed_color: discord.Color
    webhook_id: int
    prefixes: list[str]
    exit_code: int
    loading: bool
    started: datetime.datetime
    renderer: Renderer
    handlers: VariantHandlers
    def __init__(self, *args, cogs: list[str], embed_color: discord.Color, webhook_id: int, prefixes: list[str], exit_code: int = 0, **kwargs):...
    async def get_context(self, message: discord.Message) -> Coroutine[Any, Any, Context]:...
