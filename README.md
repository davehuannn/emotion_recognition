# Emotion Recognition with CNN on FER2013

This project implements a **Convolutional Neural Network (CNN)** for emotion recognition using the [FER2013 facial expression dataset](https://www.kaggle.com/datasets/msambare/fer2013). The model classifies grayscale facial images into seven emotion categories: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.

## 🚀 Features

- **Deep Learning Model:** Uses Keras and TensorFlow to build a robust CNN for image classification.
- **Data Preprocessing:** Handles reshaping, normalization, and one-hot encoding of labels.
- **Training & Evaluation:** Includes early stopping, per-class accuracy, and overall test accuracy.
- **Visualization:** Randomly displays test images with predicted and true emotions.
- **Interactive Notebook:** All code is in a Jupyter/Colab notebook for easy experimentation.

---

## 📊 Demo

![Sample Prediction](https://github.com/yourusername/yourrepo/raw/main/sample_prediction.png)
*Above: Example of a test image with predicted and true emotion labels.*

---

## 🗂️ Dataset

- **FER2013**: [Download from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- Preprocessed CSVs required:  
  - `fer2013_training_onehot.csv`
  - `fer2013_publictest_onehot.csv`

---

## 🛠️ Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo
   ```

2. **Install dependencies:**
   - If running locally:
     ```bash
     pip install tensorflow numpy matplotlib
     ```
   - Or open the notebook in [Google Colab](https://colab.research.google.com/) (recommended for GPU support).

3. **Download the dataset:**
   - Place the CSV files in your Google Drive or local directory as referenced in the notebook.

---

## 📒 Usage

1. **Open the notebook:**
   - `csci219_project_v1.ipynb` (rename as needed)

2. **Run the cells step by step:**
   - Mount Google Drive (if using Colab)
   - Load and preprocess the data
   - Build and train the CNN model
   - Evaluate accuracy and visualize predictions

3. **Try it interactively:**
   - Change the random seed to see different test images and predictions.
   - Modify the model architecture or training parameters to experiment with performance.

---

## 🧑‍💻 Example: Predicting Emotions

```python
# Select a random test image
img_indx = np.uint32(np.random.rand() * (testingset.shape[0] - 1))
sample = x_testing[img_indx, :]
sample = sample.reshape(48, 48)

# Predict class
pred_prob = model.predict(sample.reshape(1, 48, 48, 1))
pred_cls = np.argmax(pred_prob, axis=1)

plt.imshow(sample, cmap='gray')
plt.show()
print('True emotion:', get_emotion(y_testing[img_indx, :]))
print('Predicted emotion:', get_emotion(int(pred_cls[0])))
```

---

## 📈 Results

- **Overall Test Accuracy:** ~62%
- **Per-Class Accuracy:**
  - Angry: 54.8%
  - Disgust: 53.6%
  - Fear: 43.8%
  - Happy: 84.4%
  - Sad: 47.8%
  - Surprise: 79.8%
  - Neutral: 52.7%

---

## 🤝 Contributing

Pull requests and suggestions are welcome!  
Feel free to fork this repo and experiment with different architectures or datasets.

