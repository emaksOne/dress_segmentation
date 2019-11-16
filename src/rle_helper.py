import numpy as np

def rle_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return ' '.join(str(x) for x in rle)
 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
	
def test_rle_encode():
    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]]).T

    assert rle_encode(test_mask) == '7 2 11 3'
    test_mask = np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]]).T
    assert rle_encode(test_mask) == '7 2 11 3 16 1'
    test_mask = np.asarray([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]]).T
    assert rle_encode(test_mask) == '1 1 7 2 11 3'
    test_mask = np.asarray([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]]).T
    assert rle_encode(test_mask) == '1 1 7 2 11 3 16 1'

    test_mask = np.asarray([[1, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]]).T
    assert rle_encode(test_mask) == '1 1 3 2 15 2 19 3'

def test_rle_decode():
    test_rle = '7 2 11 3'
    shape = (4,4)
    assert (rle_decode(test_rle, shape) == np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]])).all() == True
    test_rle = '7 2 11 3 16 1'
    assert (rle_decode(test_rle, shape) == np.asarray([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]])).all() == True
    test_rle = '1 1 7 2 11 3'
    assert (rle_decode(test_rle, shape) == np.asarray([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]])).all() == True
    test_rle = '1 1 7 2 11 3 16 1'
    assert (rle_decode(test_rle, shape) == np.asarray([[1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]])).all() == True
    test_rle = '1 1 3 2 15 2 19 3'
    shape = (6,4)
    assert (rle_decode(test_rle, shape) == np.asarray([[1, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 0, 0]])).all() == True
