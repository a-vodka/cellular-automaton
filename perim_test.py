    import numpy as np
    import skimage.measure
    import skimage.morphology
    import skimage.filters

    from scipy import ndimage as ndi


    def vdk_perimeter(image):
        (w, h) = image.shape
        image = image.astype(np.uint8)
        data = np.zeros((w + 2, h + 2), dtype=image.dtype)
        data[1:-1, 1:-1] = image
        dilat = skimage.morphology.binary_dilation(data)
        newdata = dilat - data

        kernel = np.array([[10, 2, 10],
                           [2, 1, 2],
                           [10, 2, 10]])

        T = skimage.filters.edges.convolve(newdata, kernel, mode='constant', cval=0)

        cat_a = np.array([5, 15, 7, 25, 27, 17])
        cat_b = np.array([21, 33])
        cat_c = np.array([13, 23])
        cat_a_num = np.count_nonzero(np.isin(T, cat_a))
        cat_b_num = np.count_nonzero(np.isin(T, cat_b))
        cat_c_num = np.count_nonzero(np.isin(T, cat_c))

        perim = cat_a_num + cat_b_num * np.sqrt(2.) + cat_c_num * (1. + np.sqrt(2.)) / 2.

        return perim


    def new_std_perimeter(image, neighbourhood=4):
        STREL_4 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=np.uint8)
        STREL_8 = np.ones((3, 3), dtype=np.uint8)

        if neighbourhood == 4:
            strel = STREL_4
        else:
            strel = STREL_8
        image = image.astype(np.uint8)

        (w, h) = image.shape
        data = np.zeros((w + 2, h + 2), dtype=image.dtype)
        data[1:-1, 1:-1] = image
        image = data

        eroded_image = ndi.binary_dilation(image, strel, border_value=0)
        border_image = eroded_image - image

        perimeter_weights = np.zeros(50, dtype=np.double)
        perimeter_weights[[5, 7, 15, 17, 25, 27]] = 1
        perimeter_weights[[21, 33]] = np.sqrt(2)
        perimeter_weights[[13, 23]] = (1 + np.sqrt(2)) / 2

        perimeter_image = ndi.convolve(border_image, np.array([[10, 2, 10],
                                                               [2, 1, 2],
                                                               [10, 2, 10]]),
                                       mode='constant', cval=0)

        # You can also write
        # return perimeter_weights[perimeter_image].sum()
        # but that was measured as taking much longer than bincount + np.dot (5x
        # as much time)
        perimeter_histogram = np.bincount(perimeter_image.ravel(), minlength=50)
        total_perimeter = np.dot(perimeter_histogram, perimeter_weights)
        return total_perimeter


    image = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 1, 1, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 1],
                      [0, 0, 1, 0, 0, 1],
                      [0, 1, 1, 1, 0, 0],
                      [1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0]])

    label_img = skimage.measure.label(image)
    regions = skimage.measure.regionprops(label_img)

    exact_values = {5: 12, 3: 8, 13: 20}

    print 'area', 'standard_function', 'standard_with_modific', 'rewrited_standard', 'exact_value'
    for props in regions:
        print props.area, props.perimeter, new_std_perimeter(props.convex_image), vdk_perimeter(
            props.convex_image), exact_values[props.area]
