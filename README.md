# Tileset Builder
Use Retro Diffusion's API to create tilesets from two textures

![Gl3BUtGW8AAsFon](https://github.com/user-attachments/assets/9009f787-eb36-44f5-abd1-88b7d6172cac)


# Installation
To use this, you must have python 3.11 installed, and run `pip install requirements.txt` from the 'util' folder.
Then you can start the python script, and it will open a gradio ui in your default browser.

# Usage
You will need an API key from [https://www.retrodiffusion.ai/](https://www.retrodiffusion.ai/). You can enter this in the gradio ui, or you can save it to the util folder in a .txt file names "api_key.txt".

Now you can add the texture files you want to use (they must be the same size, and larger than 16x16 and smaller than 128x128). You can even use Retro Diffusion to generate these if you want.
You don't need to use the outside and inside prompts, but it is a good idea to get the best results possible.

Choose the "Master Mask" you want to use. This mask determines the shapes of the tiles in the tileset. There are a few defaults to choose from.

# Adding Master Masks
The master masks must be in a sepecific format and arrangement. Follow the rules below or you'll get errors and deformed tilesets:
- Colors must be pure black (0,0,0), pure white (255, 255, 255), or pure magenta (255, 0, 255).
- Black is the "outside" color.
- White is the "inside" color.
- Magenta is the "background" color. See the "platformer" tileset mask for an example of its use.
- The top left tile must be a sold black tile the full size of all the tiles in the set, and it must not have other pieces of black next to it. The program uses this tile as a reference point to determine the size of the sheet.
  ![image](https://github.com/user-attachments/assets/39de9a2f-67c7-4a27-a38c-a51e5769350f)

- The sheet must be a size multiple of the top left tile.
- You can have empty tiles, simply make the whole tile solid black, solid white, or solid magenta.
