# Job-offer-predictor
AI Model that predicts whether the students got a job offer or not based on skills, projects, etc.
First: Dataset Overview
The dataset used in this study comprises information from 20,000 students who participated in a job fair. It contains six key features that represent various aspects of a student's academic and extracurricular profile, alongside a binary label indicating whether the student received a job offer. After data cleaning, duplicates were removed, and the dataset was randomly shuffled to ensure unbiased analysis.
Each record includes:
•	skills: A semicolon-separated list of technical skills (e.g., Python, SQL, Machine Learning).
•	experience years: Number of years of relevant experience.
•	course grades: Average course grade (on a 0–100 scale).
•	projects completed: Number of academic or personal projects completed.
•	extracurriculars: Count of extracurricular activities the student engaged in.
•	job offer: Binary target variable (1 if the student received a job offer, 0 otherwise).
This clean and well-structured dataset provides a solid foundation for exploring the relationships between students’ backgrounds and their chances of receiving a job offer, which can be valuable for predictive modeling and feature importance analysis in career-related machine learning tasks.
Second: Encoding
Determining the best encoding technique for a dataset depends on:
1.	The type of categorical data (nominal vs ordinal).
2.	The number of unique categories.
3.	The algorithm you plan to use.
4.	Whether your dataset is sparse or dense.
5.	Whether there is a risk of overfitting.
6.	Interpretability and speed requirements.
How to find which encoding technique suits your dataset?
Dataset/Feature Condition	Recommended Encoding
Few unique nominal categories	One-Hot Encoding
Many unique nominal categories	Target / Binary / Frequency
Ordered categories	Ordinal Encoding or Label
Used with tree-based models	Label / Ordinal / Frequency
Used with linear or distance-based models	One-Hot or Target Encoding
High cardinality with target leakage risk	K-Fold Target or Frequency

Best Encoding Technique for Student Job Fair Dataset
Our goal is to encode the ‘skills’ column. That is why I chose the ‘MultiLabelBinarizer’ encoding technique, because the column ‘skills’ is a multi-label categorical feature, stored as a string like 'SQL; C++'.
What is MultiLabelBinarizer?
MultiLabelBinarizer is a preprocessing class from scikit-learn that is used to convert a list of multiple labels (per sample) into a binary (0/1) matrix format. It is useful when each instance (row) can have multiple categories/labels — i.e., multi-label classification.
How does it work?
MultiLabelBinarizer will:
1.	Identify all unique labels across all samples: ['C++', 'Python', 'SQL']
2.	Create binary columns (one for each label)
3.	Assign 1 or 0 depending on whether that label is present for the sample
The result is a Data Frame like this:
C++	Python	SQL
0	1	1
1	0	0
1	1	1
Use cases
MultiLabelBinarizer is ideal for:
•	Skills data like your dataset ("skills": "SQL; Python")
•	Genres in movies/music (e.g., a movie can be both 'Action' and 'Sci-Fi')
•	Tags in Stack Overflow posts (a post can have multiple tags)
•	Symptoms in medical records (a patient can have several symptoms)
Third: Feature Engineering
All numeric features are already suitable. But let us consider some derived features:
Optionally Added:
•	skill count: how many skills the student has.
Sampling
After checking the imbalance in our dataset using the imbalance ratio, the dataset was already balanced because the imbalance ratio was less than 1.5.
Scaling
After choosing the xgboost classification model, which does not need a scaling process because it saves preprocessing effort compared to logistic regression or SVM.
Fourth: Model Building
Since we are predicting a binary outcome (job offer: 0 or 1), using structured numerical and categorical (skills) features.
Let us compare models based on:
1.	Data characteristics
2.	Encoding needs
3.	Performance tendencies
4.	Interpretability
Best Model for Our Dataset: Tree-Based Models (like Random Forest or XGBoost)
Why Tree-Based Models Are Best for this dataset?
Criteria	Explanation
Manages Mixed Feature Types	Works with both numeric + one-hot encoded categorical features (skills) without scaling
Captures Non-linear Interactions	Tree models naturally learn thresholds & non-linear combinations (e.g., “if course_grade > 90 and has_ML”)
Robust to Outliers and Skew	Tree-based models do not assume normal distribution
No Need for Scaling	Saves preprocessing effort compared to logistic regression or SVM
Manages Multi-label Encoding	Sparse binary columns from skills do not harm trees like they would hurt logistic regression or KNN
Works well with Redundant Features	Feature importances can guide you to reduce noise

So, as a first step of training, I defined features and the target. Target is ‘job_offer’, and other columns are the features since they all affect the target.
Then I used the ‘train_test_split’ function to get the training and testing data.
In addition to that, I defined an xgboost classifier to train the model.
Finally, I predicted the model using ‘xtest’. Also, I printed the accuracy, confusion matrix with visualization, and classification report.
 
Figure 1: Confusion Matrix
Fifth: Features’ Importance
We are now at the feature importance step, which is all about understanding which features most influence job offer predictions. This step helps with:
•	Interpretability (why predictions are made).
•	Feature selection (remove unimportant features).
•	Strategic insights (what matters most about getting a job offer).
I used a bar plot to represent features and how important they are.
 
Figure 2: Feature Importance plot
As we can see in the bar plot. Values are all near 0.1, and the least important value is 0.08. So, they are all important, and there is no need to skip any feature while training. But the most important feature is data analysis.
Sixth: Model Evaluation
I used cross-validation to evaluate our model to show the following results:
Cross-validation scores: [0.493      0.49225    0.5025     0.49687422 0.48537134]
Also, I checked overfitting and underfitting by predicting with both ‘xtrain’ and ‘xtest’ and comparing their accuracy scores to show the following results:
Train Accuracy: 0.78, Test Accuracy: 0.50.
As we can see in the plot. There is a significant difference between training accuracy and testing accuracy. Which means there is overfitting. That can be solved in many ways, like reducing model depth, etc.
 
Figure 3: Overfitting and Underfitting plot
