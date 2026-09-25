import depthai as dai

model = dai.NNArchive("YOLOv26n_test2_model.rvc2.tar.xz")
# Alternatively, use your converted local artifact:
# model = dai.NNArchive("path/to/model.rvc4.tar.xz")

visualizer = dai.RemoteConnection()

with dai.Pipeline() as pipeline:
    camera = pipeline.create(dai.node.Camera).build()
    detection = pipeline.create(dai.node.DetectionNetwork).build(camera, model)

    visualizer.addTopic("rgb", detection.passthrough, group="RGB")
    visualizer.addTopic("detections", detection.out, group="RGB")

    pipeline.start()
    visualizer.registerPipeline(pipeline)

    while pipeline.isRunning():
        if visualizer.waitKey(1) == ord("q"):
            pipeline.stop()
