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
- cv2
- Numpy
- Keras
- datetime
- sklearn
- mlxtend
- preparation
- detection
- behavior
- expression
- plot

## Emotional patterns and meaning
| Pattern | Symbol |  Meaning |
| :-------- | :--------: | :--------- |
|Positive| + + | Easy to chew, Taste good and feeling satisfied after eating|
|Neutral | N N | Easy to chew and feeling normal after eating|
|Negative | - - | Chewing food at first and feeling less satisfied With food|
|Positive Neutral | + N | Easy to chew, Taste good and feeling normal after eating|
|Positive Negative | + - | Easy to chew, Taste good and feeling less satisfied With food|
|Neutral Positive | N + | Easy to chew and feeling satisfied after eating|
|Neutral Negative | N - | Easy to chew and feeling less satisfied With food|
|Negative Positive | - + | Chewing food at first and feeling satisfied after eating|
|Negative Neutral | - N | Chewing food at first and feeling normal after eating|

## Usage
To run the program, just type:
```python
python main.py --video {VIDEO_PATH}
```
Then the program will display FER graph of your video.
