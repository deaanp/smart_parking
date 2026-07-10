from roboflow import Roboflow
rf = Roboflow(api_key="U9Y9izUs6CxPBWhiaHCE")
project = rf.workspace("sttn").project("brd-ghgv6")
version = project.version(3)
dataset = version.download("yolov8")
                