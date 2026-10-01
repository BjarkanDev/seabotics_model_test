import depthai as dai

archive = dai.NNArchive("YOLO26_buoy_detector.rvc2.tar.xz")
print("Model input:", archive.getInputSize())  # Check for 416 × 416

if not archive.getConfig().model.heads:
    raise RuntimeError("Archive has no detection head for decoding YOLO outputs")

with dai.Pipeline() as pipeline:
    camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    detector = pipeline.create(dai.node.DetectionNetwork).build(camera, archive)
    detections = detector.out.createOutputQueue()

    pipeline.start()
    while pipeline.isRunning():
        message = detections.get()
        print([(d.label, d.confidence) for d in message.detections])
import depthai as dai

model = dai.NNArchive("YOLO26n_buoy_detector.rvc2.tar.xz")
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
