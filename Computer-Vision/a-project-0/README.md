# Project 0：学会使用Python

> Original Instruction:
>
> 这是我们的预备Project，帮助大家掌握Python读取与显示图像的相关用法。
>
> 1. 希望各位同学能够安装Anaconda与Pycharm，如果你选择其他IDE进行编程也可以，例如Visual Studio Code。
> 2. 熟悉Conda与Pip的使用方法，安装Pillow，OpenCV，Matplotlib，Numpy等工具包。
> 3. 自己选择至少5幅照片，通过Pillow与OpenCV进行读取，然后通过Matplotlib进行显示，观察Pillow与OpenCV读取的图像j结果有何异同点。
> 4. 采用Pillow或者OpenCV对图像放大3倍处理和沿着x方向平移2个像素处理。
> 注意事项：
> 5. 此次作业为强制性作业，目的在于帮助大家理解Python中图像读取与显示的方法；希望每位同学都能够去做一下。
> 6. 如果有问题，请在群里咨询助教。

## Code

### How to Run the Code

```sh
cd Your/Path/to/Project-0

./run.sh
```

### `read_and_display.py`

Read the 5 pictures under the folder `pics` using `pillow` and `opencv` separately, and save the pictures displayed via `matplotlib`.

- Input: `Project-0/pics`
- Output: `Project-0/results/read_and_display`

For the convenience of comparison, the `matplotlib` displaying of `pillow` load and `opencv` load are put side-by-side in one output file.

### `scale_and_move.py`

Use `pillow` to read and scale the picture to 3x, and then move the picture to the left for 2 pixels.

- Input: `Project-0/pics`
- Output: `Project-0/results/scale_and_move`

## Answer

1. `pillow`'s read would return in RGB format, while `opencv` returning in BGR format. To make the output identical, further conversion is needed.