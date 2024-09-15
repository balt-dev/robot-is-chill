import tomlkit
import glob
from pathlib import Path

from src.types import TilingMode

def main():
    for path in glob.glob("data/custom/*.toml"):
        path = Path(path)
        print(f"Reserializing {path}...")
        with open(path, "r") as file:
            tiles = {key: value for key, value in tomlkit.load(file).items()}
        
        doc = tomlkit.document()
        doc.add(tomlkit.comment("See CONTRIBUTING.md for how to properly edit this file."))
        doc.add(tomlkit.nl())
        doc.add(tomlkit.nl())
        doc.add(tomlkit.nl())

        for name, data in tiles.items():
            if data["tiling"] == +TilingMode.TILING and data.pop("diagonal"):
                data["tiling"] = +TilingMode.DIAGONAL_TILING
            data["tiling"] = str(TilingMode(data["tiling"]))

            table = tomlkit.inline_table()
            table.update(data)
            doc.add(name, table)
            doc.add(tomlkit.nl())

        toml_path = path.with_suffix(".toml")
        with open(toml_path, "w+") as file:
            tomlkit.dump(doc, file)

    print("Done.")


if __name__ == "__main__":
    main()