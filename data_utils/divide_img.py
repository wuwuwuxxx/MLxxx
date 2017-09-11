def divide_img(imgs, cropsize):
    """
    :param imgs: list of images
    :param cropsize: int
    :return: divided small images
    """
    result = list()
    for img in imgs:
        shape = img.shape
        if len(shape) == 3:
            h, w, _ = shape
        elif len(shape) == 2:
            h, w = shape
        for row in range(0, h, cropsize):
            for col in range(0, w, cropsize):
                y = row if row + cropsize <= h else h - cropsize - 1
                x = col if col + cropsize <= w else w - cropsize - 1
                if len(shape) == 3:
                    result.append(img[y:y + cropsize, x:x + cropsize, :])
                elif len(shape) == 2:
                    result.append(img[y:y + cropsize, x:x + cropsize])
    return result
