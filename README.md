# Gauge Reader

Welcome to my learnings with Python's computer vision library! The problem I am trying to solve here? Take a photograph of an analog gauge and return the value corresponding to the needle's current position. This relies on a few assumptions about said gauge which are cited below.

## Using this code

You can save the files wherever you want on disk. Just download the `requirements.txt` and you can execute the main program like so:

```bash
./main <image_name> <output_image> --verbose
```

where `image_name` and `output_image` represent where the input image will come from and where the output image will be saved to. The `--verbose` flag is optional and significantly slows th program. It takes a break at each step of image processing to show the image to you on the screen waiting for keyboard input to proceed. It also prints out a bunch of intermediary calculations and findings.

You can also remove my sample images from the images folder and replace them with images of your own. Or test with my sample images provided. The best way to test them all out is using the `./process_images.sh` bash script which just iterates over all files in the `/images` folder, processes them, and saves the output to the `/output` folder under the same name. It should work with your own images again assuming they conform to the above assumptions.

### Assumptions

1. The photo passed containing the gauge is taken head-on (not at an angle or rotated at all) and the gauge itself takes up at least half of the picture but not more than the boundary of the picture.
2. The gauges values scale linearly from the min to the max value (not logarithmically or something else).
3. The needle for the gauge has 2 sides to it that are not parallel but rather come to a point.
4. There is only one needle on the gauge.

and I think that's it... Oh yeah also I hard coded everything for now. I will go back and clean it up later. Probably...

### Other things

I found out by doing this project that I did not pay very close attention in trigonometry during high school XD. If you did and you see things that can be improved with my mathematics then feel free to suggest those things. I may change them if I understand your suggestion and I may not.

The OpenCV library uses BGR instead of RGB to encode images in numpy arrays, and the arrangement of pixels is from the perspective of top left to bottom right (as if you were reading a book) instead of how human eyes work where down is down and up is up for a picture. So it is all flipped which is fun for the math parts.

And always remember to have fun!

### Other other things

I started reorganizing the code a little. The `./process_images.sh` should still work, but there are some other files out there right now that are lkind of irrelevant so feel free to ignore them.
