from interpolation import interpolation
from affineTransformation import affine
from hist_eq import hist_eq
from hist_spec import hist_spec
from boxfilter import boxfilter
from edgedetection import edgedetection
from unsharpmask import unsharpmask
from meanfilters import meanfilter
from orderfilter import orderfilter
from dct_proj import dct_proj

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
    if x == 9:
        dct_proj()



if __name__ == '__main__':
    # x = int(input('1: Interpolation\n'
    #               '2: Affine Transformation\n'
    #               '3: Histogram Equalization\n'
    #               '4: Edge Detection\n'
    #               '5: Unsharp Masking and Highboost Filter\n'
    #               '6: Boxfilter\n'
    #               '7: Meanfilter\n'
    #               '8: Order-statistic Filter\n'
    #               '9: The Final Projects\n'
    #               'Enter Example Number(1 ~ 9) : '))
    # while (x < 10):
    #     sel_run(x)
    #     print('Press Ctrl+C to Exit.')
    #     x = int(input('Enter Example Number(1 ~ 9) : '))
    sel_run(9)
    exit()