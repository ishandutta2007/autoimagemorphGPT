from mypackage import run_autoimagemorphGPT

def test_run_autoimagemorph():
    # python autoimagemorphGPT.py -inframes "['f0.png','f30.png','f60.png','f90.png','f120.png','f150.png','f0.png']" -outprefix f -framerate 30 -subpixel 4
    run_autoimagemorphGPT(inframes="['f0.png','f30.png','f60.png','f90.png','f120.png','f150.png','f0.png']", outprefix="f", framerate=30, subpixel=4)
