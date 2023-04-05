from interpolation import interpolation
from affineTransformation import affine
from hist_eq import hist_eq
from hist_spec import hist_spec
from boxfilter import boxfilter
from edgedetection import edgedetection
from unsharpmask import unsharpmask
from meanfilters import meanfilter
from orderfilter import orderfilter


def run():
    interpolation()
    affine()
    hist_eq()
    hist_spec()
    boxfilter()
    edgedetection()
    unsharpmask()


def sel_run(x):
    if x == 1:
        interpolation()
    if x == 2:
        affine()
    if x == 3:
        hist_eq()
    if x == 4:
        edgedetection()
    if x == 5:
        unsharpmask()
    if x == 6:
        boxfilter()
    if x == 7:
        meanfilter()
    if x == 8:
        orderfilter()


if __name__ == '__main__':
    x = int(input('Enter Example Number(1 ~ 9) : '))
    while (x < 10):
        sel_run(x)
        print('Press Ctrl+C to Exit.')
        x = int(input('Enter Example Number(1 ~ 9) : '))
    exit()