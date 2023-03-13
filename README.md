# Robot Is Chill

---

This is a self-sustained fork of "Robot is you" project of RocketRace,
if you want to see the original project, go here:
https://github.com/RocketRace/robot-is-you#readme

[Support Server](https://discord.gg/ktk8XkAfGD)

---

## Differences:
* More sprites!
* More filters!
* More commands!

---

### Setup
- Clone the repository
- `pip install -r requirements.txt`
- Set up files
  - auth.py: 
  ```py
  token: str = "<TOKEN>"
  ```
  - webhooks.py:
  ```py
  logging_id: int = <command logging id>
  error_id: int = <error logging id>
    ```
  - Make directory `target/renders/`
  - Configure `config.py`
- Run the bot
- Run setup commands (in order)

  | Command | What it does |
  | :------ | :----------- |
  | `loadbaba <path>`| Loads required assets from the game from the path. Must have a copy of the game to do this. |
  | `loaddata`| Loads tile metadata from the files. |
  | `loadworld *`| Loads all maps. |
  | `loadletters`| Slices letters from text objects for custom text. |

- Restart the bot
Everything should be working.
