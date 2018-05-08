This script creates ASCII art from image or video inputs.


##  Seeing Is Believing!
![alt text](https://github.com/scoliann/MyLifeInASCII/blob/master/readme_resources/eiffel_tower_before_after.jpg)


## Inspiration

After taking Professor Irfan Essa's Computational Photography class at Georgia Tech, I realized that I had all the necessary know-how to implement a pipeline that generates ASCII art from an input image.  I expanded on this idea, and created a flexible pipeline that enables user control of the synthesis process, and works on both video and image inputs.

## How To Use

There are two steps to utilize the media-to-ASCII pipeline:

1.  Set the values for the key-value pairs in `config.yaml` as desired.  This includes specifying the input file name.
2.  Make sure the input file is in the same directory as `media_to_ascii.py`, and run `python media_to_ascii.py`.

After completing these two steps, an output media file with `_output` in its name will be created in the same directory as the input file.

## Synthesis Parameters

All parameters affecting the synthesis process are controlled by editing the contents of `config.yaml`.  A brief explanation of the key-value pairs in this file are as follows:

1.  `img_exts` - A list of image extensions that `media_to_ascii.py` will recognize.
2.  `vid_exts` - A list of video extensions that `media_to_ascii.py` will recognize.
3.  `ascii_sprite_file` - An included image file that contains sprites for each ASCII character.  (Do not edit this).
4.  `num_ascii_chars` - The count of the number of characters in `ascii_sprite_file`.  (Do not edit this).
5.  `ascii_char_list` - The list of ASCII characters that should be used during synthesis.  If the list contains 'all', then all ASCII characters will be used.
5.  `sprite_resize` - Percent change of the ASCII sprites' dimensions.  (E.g. 1.5 == 150% the normal ASCII character sprite dimensions). 
6.  `sprite_dilations` - Number of iterations that a 2x2 kernel should be applied to "[dilate](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#dilation)" the black regions of ASCII sprites.  If negative, then the specified number of iterations are applied to "[erode](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html#erosion)" the black regions of ASCII sprites.  If set to 0, nothing happens.
6.  `sprite_color` - Boolean for whether the output artifact should be in color (True) or grayscale (False).
7.  `sprite_color_thresh` ' A threshold value for specifying what values will be considered white. vs non-white. (Do not edit this).
8.  `img_resize` - Percent change of the input media's dimensions when creating the output artifact.  (E.g. 1.5 == 150% input media dimensions).
9.  `nth_frames` - The number of frames that should be kept for video input.  For example if set to 3, then every third frame will be kept.
10. `fps` - The frames per second for the output video.
11. `input_file` - The input media file name.

## Algorithm / Pipeline

To understand the algorithm used to generate ASCII art, refer to Matthew Mikolay's [excellent post](http://mattmik.com/articles/ascii/ascii.html), as well as [this](https://stackoverflow.com/questions/394882/how-do-ascii-art-image-conversion-algorithms-work) StackOverflow post.


## Additional Features

In addition to implementing the "typical" ASCII art algorithm, this project includes some additional features:

1.  ASCII art outputs can be generated for input images or videos.
2.  The user has the option to choose between black and white, or color outputs.
3.  The user can choose to dilate or erode (as well as resize) the ASCII characters, granting excellent control over the quality of the output.
4.  The user can choose which ASCII characters should appear in the output.
5.  The video pipeline is implemented with multiprocessing to ensure fast runtimes.


## Synthesis Tips

I would feel remiss if I didn't list a few tips to aid in the synthesis process:

1.  To make finer details of an image apparent:  Increase the output image size, decrease the ASCII sprite size, or both.
2.  To increase the amount of color:  Set the `sprite_dilations` to an integer greater than 0.  This will make the ASCII characters "bolder", thereby increasing the total area of the ASCII characters and making the image more colorful.
2.  Often synthesizing in color makes finer details more apparent.


## Results

For some example results, check the `example_results` folder.

