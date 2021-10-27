from __future__ import annotations
import random

import numpy as np
from copy import *
import threading
import asyncio 

from itertools import product

from discord.ext import commands

from .. import constants, errors
from ..types import Bot, Context

from itertools import product

def solve_sudoku(size, grid):  
  '''Author: Ali Assaf <ali.assaf.mail@gmail.com>
Copyright: (C) 2010 Ali Assaf
License: GNU General Public License <http://www.gnu.org/licenses/>'''
  def exact_cover(X, Y):
      X = {j: set() for j in X}
      for i, row in Y.items():
          for j in row:
              X[j].add(i)
      return X, Y

  def solve(X, Y, solution):
      if not X:
          yield list(solution)
      else:
          c = min(X, key=lambda c: len(X[c]))
          for r in list(X[c]):
              solution.append(r)
              cols = select(X, Y, r)
              for s in solve(X, Y, solution):
                  yield s
              deselect(X, Y, r, cols)
              solution.pop()

  def select(X, Y, r):
      cols = []
      for j in Y[r]:
          for i in X[j]:
              for k in Y[i]:
                  if k != j:
                      X[k].remove(i)
          cols.append(X.pop(j))
      return cols

  def deselect(X, Y, r, cols):
      for j in reversed(Y[r]):
          X[j] = cols.pop()
          for i in X[j]:
              for k in Y[i]:
                  if k != j:
                      X[k].add(i)
  R, C = size
  N = R * C
  X = ([("rc", rc) for rc in product(range(N), range(N))] +
    [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
    [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
    [("bn", bn) for bn in product(range(N), range(1, N + 1))])
  Y = dict()
  for r, c, n in product(range(N), range(N), range(1, N + 1)):
    b = (r // R) * R + (c // C) # Box number
    Y[(r, c, n)] = [
      ("rc", (r, c)),
      ("rn", (r, n)),
      ("cn", (c, n)),
      ("bn", (b, n))]
  X, Y = exact_cover(X, Y)
  for i, row in enumerate(grid):
    for j, n in enumerate(row):
      if n:
        select(X, Y, (i, j, n))
  for solution in solve(X, Y, []):
    for (r, c, n) in solution:
      grid[r][c] = n
    yield grid
    
def exit_after(s):
    def quit_function():
      threading.interrupt_main()
    def outer(fn):
        def inner(*args, **kwargs):
            timer = threading.Timer(s, quit_function)
            timer.start()
            try:
                result = fn(*args, **kwargs)
            finally:
                timer.cancel()
            return result
        return inner
    return outer
  
class MiscCog(commands.Cog, name="Miscellaneous Commands"):
  def __init__(self, bot: Bot):
    self.bot = bot
  @commands.command(hidden=True)
  @commands.cooldown(5, 8, type=commands.BucketType.channel)
  async def misc(self, ctx: Context, subcommand: str = None, *, args: str):
    '''Does a variety of different miscellaneous things. Subcommands:
`sudoku` - Solve a sudoku. Inputs are formatted as space-separated numbers, or a dash for blank. 
Example:
`1 - 3 - 5 - 7 - 9`
`4 - - 7 - - 1 - -`
         `⁝`'''
    if not subcommand:
      return await ctx.error('Subcommand not specified!')
    await ctx.trigger_typing()
    if subcommand == 'sudoku':
      board=[]
      for t in args.split('\n'):
        try:
          board.append([int(n) if n != '-' else 0 for n in t.split(' ')])
        except:
          return await ctx.error(f'Invalid input!')
      try:
        assert len(board) == 9 and len(board[0]) == 9
      except:
        return await ctx.error(f'Invalid input!')
      solved = list(solve_sudoku((3,3),board))
      if len(solved) == 0:
        return await ctx.send(f'Sudoku is impossible.')
      random.shuffle(solved)
      printer = []
      for y in solved[0]:
        printer.append(' '.join([str(x) for x in y]))
      printer="\n".join(printer)
      m = ''
      if len(solved) > 1:
        m = f'*{len(solved)} solutions were found. Showing random solution...*\n'
      return await ctx.send(f'{m}```\n{printer}```')
def setup(bot: Bot):
    bot.add_cog(MiscCog(bot))