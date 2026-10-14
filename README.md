# face-emotion-recognition-software
Real-Time Facial Emotion Recognition System  Developed a real-time facial emotion recognition system using Python, TensorFlow/Keras, and OpenCV. The project uses a Convolutional Neural Network (CNN) trained to classify seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise
CNN algorithm use
(convolution neural network)
images                            label
images/angry/photo1.png           angry
images/angry/photo2.png           angry
images/sad/photo1.png              sad
images/sad/photo2.png              sad
and many more..
inka use kr ke table bne ga pandas model use kr ke ek ek image ko process karenge 
inko ek array mai convert krna hai jiska method hai tensor flow mai load images
is method mai ek ek kr ke image denge and image in gray scale

load_img(img,grayscale=True)

load image ka jo output hoga vo usko convert krke store kr lenge ek jagah model ko train krne ke liye
image ko process krne ke baad model banana hai jisme bohot saari layers hongi
like 2-d layer etc.

