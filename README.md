# Underwater-object-identification

- Objectives:
  - To design deep learning model to detect trash in underwater environment.
  - To test images collected from rivers/streams in Norfolk on the model.
- Software:
  - Pytorch using Google colab
- Setup procedure:
  - Mount drive on Google colab
  - Import required python files to the colab environment
  - Run codes on cuda and select hyperparameters
  - Get graphs for key metrics in object detection.
- Results (Major findings):
  - An accuracy of 0.9 was achieved on the test dataset.
  - Other key metrics like precision and recall were strong indicating that the model is robust.
  - The model was able to detect the images collected from rivers/streams in Norfolk by the BlueROV2 robot but the iou was low compared to that of the test dataset imported from J-EDI marine debris database.
