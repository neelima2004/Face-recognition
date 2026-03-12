# Face-recognition
Develop an accurate face recognition system using modern face embeddings


# Face Recognition System

A deep learning–based **Face Recognition System** developed during my internship at **HMI Solutions** as part of the requirements for **B.Tech in Computer Science and Engineering**.

This project uses **128-dimensional facial embeddings** instead of traditional feature descriptors (ORB) and classifiers (KNN/SVM). The system identifies faces by comparing embeddings using **Euclidean distance**, providing higher accuracy and robustness.

---

# Project Overview

Traditional face recognition approaches struggle with variations in:

* Lighting conditions
* Facial expressions
* Pose variations
* Occlusions (glasses, masks, hats)

To overcome these limitations, this project uses the **face_recognition library (built on dlib)** to generate deep facial embeddings and match faces efficiently.

---

# Features

* Deep learning–based **face embeddings**
* **128-dimensional facial encoding**
* **Euclidean distance matching**
* No need for classifiers like **KNN or SVM**
* Supports **multiple known faces**
* Easily extendable by adding new images

---

# Technologies Used

| Technology       | Purpose                                   |
| ---------------- | ----------------------------------------- |
| Python           | Core programming                          |
| OpenCV           | Image processing & drawing bounding boxes |
| face_recognition | Face detection and embedding generation   |
| NumPy            | Numerical computations                    |
| Pickle           | Saving and loading label dictionaries     |

---

# Project Workflow

## 1. Training Phase

* Load known images
* Detect faces
* Generate **128-D facial encodings**
* Store encodings and labels

Example:

```
known_face_encodings = [
 [0.1, 0.2, 0.3],
 [0.5, 0.4, 0.6],
 [0.9, 0.8, 0.95]
]

label_dict = {
 0: "Person A",
 1: "Person B",
 2: "Person C"
}
```

Encodings are saved using:

```
np.save("face_encodings.npy", known_face_encodings)
pickle.dump(label_dict, open("label_dict.pkl", "wb"))
```

---

## 2. Testing Phase

* Capture a new image
* Detect faces
* Extract encoding
* Compare with stored encodings using **Euclidean distance**
* Identify the closest match

Example:

```
distances = [0.024, 0.58, 1.1]
best_match_index = np.argmin(distances)
name = label_dict[best_match_index]
```

Output example:

```
["Alice", "Unknown"]
```

---

# Algorithm Steps

1. Convert image from **BGR → RGB**
2. Detect face locations
3. Generate face encodings
4. Compare with known encodings
5. Find the closest match
6. Retrieve the label
7. Draw bounding box and name on the image
8. Return recognized faces

---

# Applications

* Attendance systems (Schools & Offices)
* Security and access control
* Photo tagging systems
* Surveillance systems
* Airport identity verification

---

# Future Improvements

* Use **MTCNN / RetinaFace** for better face detection
* Implement **real-time webcam recognition**
* Build a **Flask or Django API**
* Store face data in **SQL / NoSQL databases**
* Integrate with **cloud platforms**

---

# References

* face_recognition Library
  [https://github.com/ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)

* Dlib Facial Recognition
  [http://dlib.net/](http://dlib.net/)

* FaceNet Paper
  [https://arxiv.org/abs/1503.03832](https://arxiv.org/abs/1503.03832)

* OpenCV Documentation
  [https://docs.opencv.org/](https://docs.opencv.org/)

* NumPy Documentation
  [https://numpy.org/doc/](https://numpy.org/doc/)

---
