
# AI Tools for Software Engineering - Week 3 Summary

## Project Overview
This project demonstrates the versatility and practical applications of AI tools in software engineering through three interconnected tasks. Each task explores a different paradigm of machine learning, progressing from traditional algorithms to modern deep learning and natural language processing approaches.

## Project Significance
The project showcases how different AI techniques can be applied to solve diverse software engineering challenges:
- Classical ML for structured data analysis
- Deep Learning for computer vision
- NLP for text processing and understanding

## Key Achievements

---

### Task 1: Classical ML - Iris Classification
**Achievement**: Successfully implemented a Decision Tree Classifier achieving 93% accuracy on the Iris dataset.

**Implementation Highlights**:
```python
# Model Training
clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train, y_train)

# Evaluation
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Visualization
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=x.columns,
    class_names=le.classes_,
    filled=True,
    rounded=True
)
```

**Key Visualizations**:

1. Decision Tree Structure  
   ![Decision Tree](decision_tree.png)

2. Model Performance  
   ```
   Classification Report:
   
              precision    recall  f1-score
   setosa        1.00      1.00      1.00
   versicolor    0.91      0.89      0.90
   virginica     0.89      0.91      0.90
   ```

**Impact**: Demonstrated practical implementation of classical ML pipeline with emphasis on model interpretability and visualization.

---

### Task 2: Deep Learning - MNIST Digit Classification

**Achievement**: Built and trained a Convolutional Neural Network (CNN) on the MNIST dataset, achieving high accuracy.

**Implementation Highlights**:
```python
# CNN Model Architecture (PyTorch)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

**Key Visualizations**:

1. Sample Images from MNIST  
   ![MNIST Samples](mnist_samples.png)

2. Class Distribution in Training Set  
   ![Class Distribution](mnist_class_distribution.png)

3. Training Progress (Loss & Accuracy)  
   ![Training Progress](training_progress.png)

4. Per-Class Test Accuracy  
   ![Per-Class Accuracy](per_class_accuracy.png)

5. Confusion Matrix  
   ![Confusion Matrix](confusion_matrix.png)

6. Sample Predictions  
   ![Sample Predictions](sample_predictions.png)

7. Misclassified Samples  
   ![Misclassified Samples](misclassified_samples.png)

8. Feature Maps (Conv1)  
   ![Feature Maps Conv1](feature_maps_conv1.png)

9. Feature Maps (Conv2)  
   ![Feature Maps Conv2](feature_maps_conv2.png)

**Impact**: Successfully demonstrated deep learning implementation for computer vision tasks with high accuracy and rich visual analysis.

---

### Task 3: NLP - Sentiment Analysis

**Achievement**: Developed an NLP pipeline for sentiment analysis on Amazon reviews.

**Implementation Highlights**:
```python
# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Model Training (example)
model = Sequential([
    Embedding(max_words, embedding_dim),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
history = model.fit(
    x_train, y_train,
    epochs=10,
    validation_split=0.2
)
```

**Key Visualizations**:

1. Review Length Distribution  
   ![Review Length Distribution](review_length_distribution.png)

2. Entity Type Distribution  
   ![Entity Type Distribution](entity_type_distribution.png)

3. Sentiment Distribution  
   ![Sentiment Distribution](sentiment_distribution.png)

4. Sentiment Score Distribution  
   ![Sentiment Score Distribution](sentiment_score_distribution.png)

5. Sentiment vs Review Length  
   ![Sentiment vs Length](sentiment_vs_length.png)

6. Feature Correlation with Sentiment  
   ![Feature Correlation](feature_correlation.png)

7. Entity Type Sentiment  
   ![Entity Type Sentiment](entity_type_sentiment.png)

8. Entity Frequency vs Sentiment  
   ![Entity Frequency vs Sentiment](entity_frequency_vs_sentiment.png)

9. Sentiment Distribution by Entity Type  
   ![Sentiment Distribution by Entity Type](sentiment_distribution_by_entity_type.png)

10. Sentiment Summary  
    ![Sentiment Summary](sentiment_summary.png)

**Impact**: Demonstrated effective NLP implementation for real-world sentiment analysis with high accuracy and interpretable results.

---

## Technical Implementation

### Development Stack
- **Languages & Frameworks**: Python, scikit-learn, PyTorch, NLTK, spaCy, matplotlib
- **Development Tools**: Jupyter Notebooks, Git version control
- **Data Storage**: Structured datasets in `/data/` directory
- **Dependencies**: Core ML libraries and utilities

### Project Structure
```
.
├── task_1.ipynb      # Classical ML implementation
├── task_2.ipynb      # Deep Learning with MNIST
├── task_3.ipynb      # NLP Sentiment Analysis
├── mnist_samples.png
├── ...
└── data/
    ├── MNIST/       # Digit recognition dataset
    └── amazon/      # Amazon reviews dataset
```

## Key Learnings
### Technical Insights
1. **ML Pipeline Design**
   - Efficient data preprocessing strategies
   - Model selection and optimization
   - Performance evaluation techniques

2. **Deep Learning Practices**
   - CNN architecture optimization
   - Training process management
   - Resource utilization

3. **NLP Implementation**
   - Text preprocessing techniques
   - Sentiment analysis approaches
   - Scalable solution design

## Project Impact

### Achievements
1. **Performance**
   - High accuracy across all tasks
   - Efficient resource utilization
   - Scalable implementations

2. **Innovation**
   - Modern ML techniques application
   - Practical problem-solving
   - Robust error handling

3. **Documentation**
   - Clear code documentation
   - Comprehensive notebooks
   - Reproducible results

## Future Enhancements

1. **Model Improvements**
   - Advanced architecture exploration
   - Hyperparameter optimization
   - Ensemble methods integration

2. **Scalability**
   - Distributed processing
   - Memory optimization
   - Batch processing implementation

3. **Feature Additions**
   - Real-time prediction API
   - Model monitoring system
   - Automated testing pipeline

## Conclusion

This project successfully demonstrates the practical application of various AI techniques in software engineering. Through three distinct tasks, it showcases the versatility of machine learning approaches in solving different types of problems. The implementation provides a solid foundation for future AI-driven software engineering projects.