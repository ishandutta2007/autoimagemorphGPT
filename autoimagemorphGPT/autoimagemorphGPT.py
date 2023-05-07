#   Additional features :
#    - automatic triangle points selection by cv2.goodFeaturesToTrack()
#    - No GUI, single file program
#    - batch processing: transition between many images, not just 2
#    - optional subpixel processing to fix image artifacts
#    - automatic image dimensions safety (the dimensions of the first image defines the output)
#
#   Recommended postprocessing:
#    Install FFmpeg https://ffmpeg.org/
#    Example from command line:
#     ffmpeg -framerate 15 -i image%d.png output.avi
#     ffmpeg -framerate 15 -i image%d.png output.gif
#
#   TODOs:
#    - testing, error checks, sanity checks
#    - speed optimization in interpolate_points()
#    - RGBA support, currently it's only RGB
#    - tuning the parameters of cv2.goodFeaturesToTrack() in autofeaturepoints() / giving user control
#    - built-in video output with cv2 ?
#    - image scaling uses cv2.INTER_CUBIC ; tuning / giving user control ?
#    - LinAlgError ? Image dimensions should be even numbers?
#       related: https://stackoverflow.com/questions/44305456/why-am-i-getting-linalgerror-singular-matrix-from-grangercausalitytests
#         File "batchautomorph.py", line 151, in interpolate_points
#           righth = np.linalg.solve(temp_right_matrix, target_vertices)
#         File "<__array_function__ internals>", line 5, in solve
#         File "numpy\linalg\linalg.py",
#           line 403, in solve
#           r = gufunc(a, b, signature=signature, extobj=extobj)
#         File "numpy\linalg\linalg.py",
#           line 97, in _raise_linalgerror_singular
#           raise LinAlgError("Singular matrix")
#           numpy.linalg.LinAlgError: Singular matrix

def run_autoimagemorphGPT():
    import cv2, time, argparse, ast
    from scipy.ndimage import median_filter
    from scipy.spatial import Delaunay
    from scipy.interpolate import RectBivariateSpline
    from matplotlib.path import Path
    import numpy as np

    image_dir = "/Users/ishandutta2007/Documents/Projects/autoimagemorphGPT/autoimagemorphGPT/images"


    def load_triangles(limg, rimg, feature_grid_size, show_features) -> tuple:
        left_tri_list = []
        right_tri_list = []

        lr_lists = autofeaturepoints(limg, rimg, feature_grid_size, show_features)

        left_array = np.array(lr_lists[0], np.float64)
        right_array = np.array(lr_lists[1], np.float64)
        delaunay_tri = Delaunay(left_array)

        left_np = left_array[delaunay_tri.simplices]
        right_np = right_array[delaunay_tri.simplices]

        for x, y in zip(left_np, right_np):
            left_tri_list.append(Triangle(x))
            right_tri_list.append(Triangle(y))

        return left_tri_list, right_tri_list


    class Triangle:
        def __init__(self, vertices):
            if isinstance(vertices, np.ndarray) == 0:
                raise ValueError("Input argument is not of type np.array.")
            if vertices.shape != (3, 2):
                raise ValueError("Input argument does not have the expected dimensions.")
            if vertices.dtype != np.float64:
                raise ValueError("Input argument is not of type float64.")
            self.vertices = vertices
            self.min_x = int(self.vertices[:, 0].min())
            self.max_x = int(self.vertices[:, 0].max())
            self.min_y = int(self.vertices[:, 1].min())
            self.max_y = int(self.vertices[:, 1].max())

        def get_points(self):
            xList = range(self.min_x, self.max_x + 1)
            yList = range(self.min_y, self.max_y + 1)
            empty_list = list((x, y) for x in xList for y in yList)

            points = np.array(empty_list, np.float64)
            p = Path(self.vertices)
            grid = p.contains_points(points)
            mask = grid.reshape(self.max_x - self.min_x + 1, self.max_y - self.min_y + 1)

            trueArray = np.where(np.array(mask) == True)
            coordArray = np.vstack(
                (
                    trueArray[0] + self.min_x,
                    trueArray[1] + self.min_y,
                    np.ones(trueArray[0].shape[0]),
                )
            )

            return coordArray


    class Morpher:
        def __init__(self, left_image, left_triangles, right_image, right_triangles):
            if type(left_image) != np.ndarray:
                raise TypeError("Input left_image is not an np.ndarray")
            if left_image.dtype != np.uint8:
                raise TypeError("Input left_image is not of type np.uint8")
            if type(right_image) != np.ndarray:
                raise TypeError("Input right_image is not an np.ndarray")
            if right_image.dtype != np.uint8:
                raise TypeError("Input right_image is not of type np.uint8")
            if type(left_triangles) != list:
                raise TypeError("Input left_triangles is not of type List")
            for j in left_triangles:
                if isinstance(j, Triangle) == 0:
                    raise TypeError(
                        "Element of input left_triangles is not of Class Triangle"
                    )
            if type(right_triangles) != list:
                raise TypeError("Input left_triangles is not of type List")
            for k in right_triangles:
                if isinstance(k, Triangle) == 0:
                    raise TypeError(
                        "Element of input right_triangles is not of Class Triangle"
                    )
            self.left_image = np.ndarray.copy(left_image)
            self.left_triangles = left_triangles  # Not of type np.uint8
            self.right_image = np.ndarray.copy(right_image)
            self.right_triangles = right_triangles  # Not of type np.uint8
            self.left_interpolation = RectBivariateSpline(
                np.arange(self.left_image.shape[0]),
                np.arange(self.left_image.shape[1]),
                self.left_image,
            )  # , kx=2, ky=2)
            self.right_interpolation = RectBivariateSpline(
                np.arange(self.right_image.shape[0]),
                np.arange(self.right_image.shape[1]),
                self.right_image,
            )  # , kx=2, ky=2)

        def get_image_at_alpha(self, alpha, smoothMode):
            for left_triangle, right_triangle in zip(
                self.left_triangles, self.right_triangles
            ):
                self.interpolate_points(left_triangle, right_triangle, alpha)
                # print(".", end="") # TODO: this doesn't work as intended

            blend_arr = (1 - alpha) * self.left_image + alpha * self.right_image
            blend_arr = blend_arr.astype(np.uint8)
            return blend_arr

        def interpolate_points(self, left_triangle, right_triangle, alpha):
            target_triangle = Triangle(
                left_triangle.vertices
                + (right_triangle.vertices - left_triangle.vertices) * alpha
            )
            target_vertices = target_triangle.vertices.reshape(6, 1)
            temp_left_matrix = np.array(
                [
                    [
                        left_triangle.vertices[0][0],
                        left_triangle.vertices[0][1],
                        1,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        left_triangle.vertices[0][0],
                        left_triangle.vertices[0][1],
                        1,
                    ],
                    [
                        left_triangle.vertices[1][0],
                        left_triangle.vertices[1][1],
                        1,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        left_triangle.vertices[1][0],
                        left_triangle.vertices[1][1],
                        1,
                    ],
                    [
                        left_triangle.vertices[2][0],
                        left_triangle.vertices[2][1],
                        1,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        left_triangle.vertices[2][0],
                        left_triangle.vertices[2][1],
                        1,
                    ],
                ]
            )
            temp_right_matrix = np.array(
                [
                    [
                        right_triangle.vertices[0][0],
                        right_triangle.vertices[0][1],
                        1,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        right_triangle.vertices[0][0],
                        right_triangle.vertices[0][1],
                        1,
                    ],
                    [
                        right_triangle.vertices[1][0],
                        right_triangle.vertices[1][1],
                        1,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        right_triangle.vertices[1][0],
                        right_triangle.vertices[1][1],
                        1,
                    ],
                    [
                        right_triangle.vertices[2][0],
                        right_triangle.vertices[2][1],
                        1,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        right_triangle.vertices[2][0],
                        right_triangle.vertices[2][1],
                        1,
                    ],
                ]
            )
            lefth = np.linalg.solve(temp_left_matrix, target_vertices)
            righth = np.linalg.solve(temp_right_matrix, target_vertices)
            leftH = np.array(
                [
                    [lefth[0][0], lefth[1][0], lefth[2][0]],
                    [lefth[3][0], lefth[4][0], lefth[5][0]],
                    [0, 0, 1],
                ]
            )
            righth = np.array(
                [
                    [righth[0][0], righth[1][0], righth[2][0]],
                    [righth[3][0], righth[4][0], righth[5][0]],
                    [0, 0, 1],
                ]
            )
            leftinvh = np.linalg.inv(leftH)
            rightinvh = np.linalg.inv(righth)
            target_points = target_triangle.get_points()  # TODO: ~ 17-18% of runtime

            left_source_points = np.transpose(np.matmul(leftinvh, target_points))
            right_source_points = np.transpose(np.matmul(rightinvh, target_points))
            target_points = np.transpose(target_points)

            for x, y, z in zip(
                target_points, left_source_points, right_source_points
            ):  # TODO: ~ 53% of runtime
                self.left_image[int(x[1])][int(x[0])] = self.left_interpolation(y[1], y[0])
                self.right_image[int(x[1])][int(x[0])] = self.right_interpolation(
                    z[1], z[0]
                )


    # Automatic feature points
    def autofeaturepoints(leimg, riimg, featuregridsize, showfeatures):
        result = [[], []]
        for idx, img in enumerate([leimg, riimg]):
            try:

                if showfeatures:
                    print(img.shape)

                # add the 4 corners to result
                result[idx] = [
                    [0, 0],
                    [(img.shape[1] - 1), 0],
                    [0, (img.shape[0] - 1)],
                    [(img.shape[1] - 1), (img.shape[0] - 1)],
                ]

                h = int(img.shape[0] / featuregridsize) - 1
                w = int(img.shape[1] / featuregridsize) - 1

                for i in range(0, featuregridsize):
                    for j in range(0, featuregridsize):

                        # crop to a small part of the image and find 1 feature pont or middle point
                        crop_img = img[(j * h) : (j * h) + h, (i * w) : (i * w) + w]
                        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        featurepoints = cv2.goodFeaturesToTrack(
                            gray, 1, 0.1, 10
                        )  # TODO: parameters can be tuned
                        if featurepoints is None:
                            featurepoints = [[[h / 2, w / 2]]]
                        featurepoints = np.int0(featurepoints)

                        # add feature point to result, optionally draw
                        for featurepoint in featurepoints:
                            x, y = featurepoint.ravel()
                            y = y + (j * h)
                            x = x + (i * w)
                            if showfeatures:
                                cv2.circle(img, (x, y), 3, 255, -1)
                            result[idx].append([x, y])

                # optionally draw features
                if showfeatures:
                    cv2.imshow("", img)
                    cv2.waitKey(0)

            except Exception as ex:
                print(ex)
        return result


    def initmorph(startimgpath, endingpath, featuregridsize, subpixel, showfeatures, scale):
        timerstart = time.time()

        # left image load
        print("startimgpath", startimgpath)
        print("endingpath", endingpath)
        left_image_raw = cv2.imread(startimgpath)
        # scale image if custom scaling
        if scale != 1.0:
            left_image_raw = cv2.resize(
                left_image_raw,
                (
                    int(left_image_raw.shape[1] * scale),
                    int(left_image_raw.shape[0] * scale),
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        # upscale image if subpixel calculation is enabled
        if subpixel > 1:
            left_image_raw = cv2.resize(
                left_image_raw,
                (left_image_raw.shape[1] * subpixel, left_image_raw.shape[0] * subpixel),
                interpolation=cv2.INTER_CUBIC,
            )

        # right image load
        right_image_raw = cv2.imread(endingpath)
        # resize image
        right_image_raw = cv2.resize(
            right_image_raw,
            (left_image_raw.shape[1], left_image_raw.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

        left_image_arr = np.asarray(left_image_raw)
        right_image_arr = np.asarray(right_image_raw)

        # autofeaturepoints() is called in load_triangles()
        triangle_tuple = load_triangles(
            left_image_raw, right_image_raw, featuregridsize, showfeatures
        )

        # Morpher objects for color layers BGR
        morphers = [
            Morpher(
                left_image_arr[:, :, 0],
                triangle_tuple[0],
                right_image_arr[:, :, 0],
                triangle_tuple[1],
            ),
            Morpher(
                left_image_arr[:, :, 1],
                triangle_tuple[0],
                right_image_arr[:, :, 1],
                triangle_tuple[1],
            ),
            Morpher(
                left_image_arr[:, :, 2],
                triangle_tuple[0],
                right_image_arr[:, :, 2],
                triangle_tuple[1],
            ),
        ]

        print(
            "\r\nSubsequence init time: "
            + "{0:.2f}".format(time.time() - timerstart)
            + " s "
        )

        return morphers


    def morphprocess(mphs, framerate, outimgprefix, subpixel, smoothing):
        global framecnt

        # frame_0 is the starting image, so framecnt = _1
        framecnt = framecnt + 1

        # loop generate morph frames and save
        for i in range(1, framerate):

            timerstart = time.time()
            alfa = i / framerate

            # image calculation and smoothing BGR
            if smoothing > 0:
                outimage = np.dstack(
                    [
                        np.array(
                            median_filter(mphs[0].get_image_at_alpha(alfa, True), smoothing)
                        ),
                        np.array(
                            median_filter(mphs[1].get_image_at_alpha(alfa, True), smoothing)
                        ),
                        np.array(
                            median_filter(mphs[2].get_image_at_alpha(alfa, True), smoothing)
                        ),
                    ]
                )
            else:
                outimage = np.dstack(
                    [
                        np.array(mphs[0].get_image_at_alpha(alfa, True)),
                        np.array(mphs[1].get_image_at_alpha(alfa, True)),
                        np.array(mphs[2].get_image_at_alpha(alfa, True)),
                    ]
                )

            # downscale image if subpixel calculation is enabled
            if subpixel > 1:
                outimage = cv2.resize(
                    outimage,
                    (int(outimage.shape[1] / subpixel), int(outimage.shape[0] / subpixel)),
                    interpolation=cv2.INTER_CUBIC,
                )

            # write file
            filename = outimgprefix + str(framecnt) + ".png"
            framecnt = framecnt + 1
            cv2.imwrite(image_dir + "/" + filename, outimage)
            timerelapsed = time.time() - timerstart
            usppx = 1000000 * timerelapsed / (outimage.shape[0] * outimage.shape[1])
            print(
                filename
                + " saved, dimensions "
                + str(outimage.shape)
                + " time: "
                + "{0:.2f}".format(timerelapsed)
                + " s ; μs/pixel: "
                + "{0:.2f}".format(usppx)
            )


    def batchmorph(
        imgs,
        featuregridsize,
        subpixel,
        showfeatures,
        framerate,
        outimgprefix,
        smoothing,
        scale,
    ):
        global framecnt
        framecnt = 0
        totaltimerstart = time.time()

        for idx in range(len(imgs) - 1):
            morphprocess(
                initmorph(
                    image_dir + "/" + imgs[idx],
                    image_dir + "/" + imgs[idx + 1],
                    featuregridsize,
                    subpixel,
                    showfeatures,
                    scale,
                ),
                framerate,
                outimgprefix,
                subpixel,
                smoothing,
            )
        print("\r\nDone. Total time: " + str(time.time() - totaltimerstart) + " s ")


    mfeaturegridsize = 7  # number of image divisions on each axis, for example 5 creates 5x5 = 25 automatic feature points + 4 corners come automatically
    mframerate = 30  # number of transition frames to render + 1 ; for example 30 renders transiton frames 1..29
    moutprefix = "f"  # output image name prefix
    framecnt = 0  # frame counter
    msubpixel = 1  # int, min: 1, max: no hard limit, but 4 should be enough
    msmoothing = 0  # median_filter smoothing
    mshowfeatures = False  # render automatically detected features
    mscale = 1.0  # image scale

    # batch morph process
    # batchmorph(["f0.png","f30.png","f60.png","f90.png","f120.png","f150.png"],mfeaturegridsize,msubpixel,mshowfeatures,mframerate,moutprefix,msmoothing)

    # CLI arguments

    margparser = argparse.ArgumentParser(
        description="Automatic Image Morphing https://github.com/jankovicsandras/autoimagemorph adapted from https://github.com/ddowd97/Morphing"
    )
    margparser.add_argument(
        "-inframes",
        default="",
        required=True,
        help="REQUIRED input filenames in a list, for example: -inframes ['f0.png','f30.png','f60.png']",
    )
    margparser.add_argument(
        "-outprefix",
        default="",
        required=True,
        help="REQUIRED output filename prefix, -outprefix f  will write/overwrite f1.png f2.png ...",
    )
    margparser.add_argument(
        "-featuregridsize",
        type=int,
        default=mfeaturegridsize,
        help="Number of image divisions on each axis, for example -featuregridsize 5 creates 25 automatic feature points. (default: %(default)s)",
    )
    margparser.add_argument(
        "-framerate",
        type=int,
        default=mframerate,
        help="Frames to render between each keyframe +1, for example -framerate 30 will render 29 frames between -inframes ['f0.png','f30.png'] (default: %(default)s)",
    )
    margparser.add_argument(
        "-subpixel",
        type=int,
        default=msubpixel,
        help="Subpixel calculation to avoid image artifacts, for example -subpixel 4 is good quality, but 16 times slower processing. (default: %(default)s)",
    )
    margparser.add_argument(
        "-smoothing",
        type=int,
        default=msmoothing,
        help="median_filter smoothing/blur to remove image artifacts, for example -smoothing 2 will blur lightly. (default: %(default)s)",
    )
    margparser.add_argument(
        "-showfeatures",
        action="store_true",
        help="Flag to render feature points, for example -showfeatures",
    )
    margparser.add_argument(
        "-scale",
        type=float,
        default=mscale,
        help="Input scaling for preview, for example -scale 0.5 will halve both width and height, processing will be approx. 4x faster. (default: %(default)s)",
    )

    args = vars(margparser.parse_args())

    # arguments sanity check TODO

    if len(args["inframes"]) < 2:
        print(
            "ERROR: command line argument -inframes must be a string array with minimum 2 elements."
        )
        print(
            "Example\r\n > python autoimagemorphGPT/autoimagemorphGPT.py -inframes ['frame0.png','frame30.png','frame60.png'] -outprefix frame "
        )
        quit()

    if len(args["outprefix"]) < 1:
        print("ERROR: -outprefix (output filename prefix) must be specified.")
        print(
            "Example\r\n > python autoimagemorphGPT/autoimagemorphGPT.py -inframes ['frame0.png','frame30.png','frame60.png'] -outprefix frame "
        )
        quit()
    print(args["inframes"])
    args["inframes"] = ast.literal_eval(args["inframes"])

    args["featuregridsize"] = int(args["featuregridsize"])

    args["subpixel"] = int(args["subpixel"])

    args["framerate"] = int(args["framerate"])

    args["smoothing"] = int(args["smoothing"])

    args["scale"] = float(args["scale"])

    print("User input: \r\n" + str(args))

    # processing

    batchmorph(
        args["inframes"],
        args["featuregridsize"],
        args["subpixel"],
        args["showfeatures"],
        args["framerate"],
        args["outprefix"],
        args["smoothing"],
        args["scale"],
    )

if __name__ == '__main__':
    run_autoimagemorphGPT()

