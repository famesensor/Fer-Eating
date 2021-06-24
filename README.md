# Facial Expression Recognition System for Food Product Testing using Deep Learning

The aim of this project is to measure the satisfaction while testing food products.

The system is applied in Deep Learning to measure and assess the emotions of the face while eating. MobileNetV2 was chosen for detecting eating behavior, with an accuracy of 95.7%. And VGG16 was chosen for recognizing facial expression, with an accuracy of 92.9%.

The satisfaction is divided into nine types, for enabling food producers to have more accurately the satisfaction that testers have with the food or product.

## Build with

- [Python](https://www.python.org/)
- [Google Colab](https://colab.research.google.com/)

## Dataset

- The **Eating Behavior Detection** dataset was collected by us, consisting of 14 university students.
- The **Facial Expression Recognition** is trained and tested on a mixture of CK+ and JAFFE datasets.

## Used libraries
- opencv
- numpy
- keras
- sklearn
- mlxtend
- matplotlib
- plotly
- pandas

## Emotional patterns and meaning
| Pattern | Symbol |  Meaning |
| :-------- | :--------: | :--------- |
|Positive| + + | Easy to chew, Taste good and feeling satisfied after eating|
|Neutral | N N | Easy to chew and feeling normal after eating|
|Negative | - - | Chewing food at first and feeling less satisfied after eating|
|Positive Neutral | + N | Easy to chew, Taste good and feeling normal after eating|
|Positive Negative | + - | Easy to chew, Taste good and feeling less satisfied after eating|
|Neutral Positive | N + | Easy to chew and feeling satisfied after eating|
|Neutral Negative | N - | Easy to chew and feeling less satisfied after eating|
|Negative Positive | - + | Chewing food at first and feeling satisfied after eating|
|Negative Neutral | - N | Chewing food at first and feeling normal after eating|

## Usage
After installed libraries, then copy downloaded model folders to **./models** directory

To run the program, use script below.
```python
python main.py --video {VIDEO_PATH}
```
And the program will display FER graph of your video.

## Model files
- [DNN face detection](https://drive.google.com/file/d/1XSPJ8AeF7-_Sycg-wXFnwpjnsesXZUs9/view?usp=sharing)
- [YOLOv4 object detection](https://drive.google.com/file/d/1J4wekLvy4xrvUrftStZ9at1MpEgLpc3R/view?usp=sharing)
- [MobileNetV2 Behavior detection](https://drive.google.com/file/d/1dWi-9_AnNjBnk7I-g7xrz3C7cO-6dMgu/view?usp=sharing)
- [VGG16 Facial expression recognition](https://drive.google.com/file/d/1Hc-yN7_uEGD1ulACzce5_RWI1yuPnEOS/view?usp=sharing)

## License
[King Mongkut’s University of Technology Thonburi](https://www.kmutt.ac.th/en/)
