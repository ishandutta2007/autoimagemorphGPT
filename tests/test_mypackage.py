from mypackage import run_autoimagemorph

def test_run_autoimagemorph():
    # inframes = "/Users/ishandutta2007/Documents/Projects/autoimagemorphGPT/images"
    # outprefix = inframes#"path/to/output/prefix"
    # run_autoimagemorph(inframes, outprefix)
    # python autoimagemorph.py -inframes "['f0.png','f30.png','f60.png','f90.png','f120.png','f150.png','f0.png']" -outprefix f -framerate 30 -subpixel 4
    run_autoimagemorph(inframes="['f0.png','f30.png','f60.png','f90.png','f120.png','f150.png','f0.png']", outprefix="f", framerate=30, subpixel=4)
