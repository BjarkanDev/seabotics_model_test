import depthai as dai

archive = dai.NNArchive("YOLO26_buoy_detector.rvc2.tar.xz")

with dai.Pipeline() as pipeline:
    camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    nn = pipeline.create(dai.node.NeuralNetwork).build(camera, archive)
    results = nn.out.createOutputQueue()

    pipeline.start()
    while pipeline.isRunning():
        print(results.get().getFirstTensor().shape)

