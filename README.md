# ROBOT IS CHILL
---

This is a self-sustained fork of the "Robot Is You" project by RocketRace.

If you want to see the original project, go here:
https://github.com/RocketRace/robot-is-you#readme

[Support Server](https://discord.gg/ktk8XkAfGD)

---

### Setup
If on Windows, set up WSL for this, or it may get a bit messy.

Windows _is_ kind of supported, but here be dragons if you do that!

Step by step:
1. Clone the repository
2. `pip install -r requirements.txt`
3. Set up auth.py: 
   ```py
   token: str = "<TOKEN>"
   ```
4. Set up webhooks.py:
   ```py
   logging_id: int = <command logging id>
   error_id: int = <error logging id>
   ```
5. Make directory `target/renders/`
6. Configure `config.py`
7. Run the bot
8. Run setup commands (in order)

   | Command | What it does |
   | :------ | :----------- |
   | `loadbaba <path>`| Loads required assets from the game from the path. Must have a copy of the game to do this. |
   | `loaddata`| Loads tile metadata from the files. |
   | `loadworld *`| Loads all maps. |
   | `loadletters`| Slices letters from text objects for custom text. |

9. Restart the bot

Everything should be working fine!
