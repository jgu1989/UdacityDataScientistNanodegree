def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    w, h = im.size
    if w <= h:
        new_size = (256, int(256 * h / w))
    else:
        new_size = (int(256 * w / h), 256)
    im_resize = im.resize(size=new_size)

    w, h = im_resize.size
    w_new = 224
    h_new = 224
    left = (w - w_new) / 2
    top = (h - h_new) / 2
    right = (w + w_new) / 2
    bottom = (h + h_new) / 2

    im_crop = im_resize.crop((left, top, right, bottom))
    pix = np.array(im_crop)
    pix_normalized = np.zeros(pix.shape)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for i in range(w_new):
        for j in range(h_new):
            pix_normalized[i, j, :] = (pix[i, j, :] / 255 - mean) / std
    #             print(pix_normalized[i,j,:])
    pix = pix_normalized.transpose((2, 1, 0))
    return pix


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


# pix = process_image('flowers/test/1/image_06743.jpg')
# imshow(pix)