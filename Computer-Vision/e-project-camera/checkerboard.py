from PIL import Image

h = 2000
hc = 10
w = 1400
wc = 7
img = Image.new("RGB", (w, h), (0, 0, 0)) # create a new 15x15 image
pixels = img.load() # create the pixel map

for i in range (w):
    for j in range(h):
        if ((i // (h // hc)) & 1) ^ ((j // (w // wc)) & 1):
            pixels[i,j] = (0, 0, 0)
        else:
            pixels[i,j] = (255, 255, 255)

img.save('./checkerboard.png')