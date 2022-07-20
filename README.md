# Underwater-object-identification

![alt text](https://github.com/CCCS-Team/Underwater-object-identification/blob/main/imag/Picture1.png)

- Objectives:
  - To design deep learning model to detect trash in underwater environment.
  - To test images collected from rivers/streams in Virginia Beach and Chesapeake in Virginia using BlueRov2 Robots.
    - [Feb. 2022 Chesapeake VA](https://www.youtube.com/watch?v=YhF9HH67f8I&t=549s) 
    - [Apr. 2022 Virginia Beach VA](https://www.youtube.com/watch?v=Ll-X1AXM-ss)
    - [June. 2021, Norfolk State University](https://www.youtube.com/watch?v=h12f7iS8UUI)
 
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
