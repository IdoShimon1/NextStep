import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Read CSV file containing user data.
df = pd.read_csv('/Users/nicolegoihman/Desktop/NextStepProject/NextStep/nn/resume_data.csv')

# Combine degree_names and major_field_of_studies into a single 'education' field.
df['education'] = df['degree_names'] + ' ' + df['major_field_of_studies']

# Convert text features into numeric indices using LabelEncoder.
le_skills = LabelEncoder()
df['skills_enc'] = le_skills.fit_transform(df['skills'])

le_positions = LabelEncoder()
df['positions_enc'] = le_positions.fit_transform(df['positions'])

le_education = LabelEncoder()
df['education_enc'] = le_education.fit_transform(df['education'])

le_job_position = LabelEncoder()
df['job_position_enc'] = le_job_position.fit_transform(df['job_position_name'])

# Define a custom Dataset class for job matching data.
class JobMatchDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Create a dictionary of features as tensors.
        features = {
            'skills': torch.tensor(row['skills_enc'], dtype=torch.long),
            'positions': torch.tensor(row['positions_enc'], dtype=torch.long),
            'education': torch.tensor(row['education_enc'], dtype=torch.long),
            'job_position': torch.tensor(row['job_position_enc'], dtype=torch.long)
        }
        # The target is the matched score (a float between 0 and 1).
        target = torch.tensor(row['matched_score'], dtype=torch.float)
        return features, target

# Split the data into training and testing sets.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = JobMatchDataset(train_df)
test_dataset = JobMatchDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the neural network model using embedding layers for each categorical feature.
class JobMatchModel(nn.Module):
    def __init__(self, num_skills, num_positions, num_education, num_job_position, embedding_dim=8):
        super(JobMatchModel, self).__init__()
        # Embedding layers for each feature.
        self.skill_embedding = nn.Embedding(num_skills, embedding_dim)
        self.position_embedding = nn.Embedding(num_positions, embedding_dim)
        self.education_embedding = nn.Embedding(num_education, embedding_dim)
        self.job_position_embedding = nn.Embedding(num_job_position, embedding_dim)
        
        # Fully connected layers after concatenating the embeddings.
        self.fc1 = nn.Linear(embedding_dim * 4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer for the matched score.
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # To ensure output is between 0 and 1.

    def forward(self, features):
        # Obtain embeddings for each feature.
        skill_emb = self.skill_embedding(features['skills'])
        position_emb = self.position_embedding(features['positions'])
        education_emb = self.education_embedding(features['education'])
        job_position_emb = self.job_position_embedding(features['job_position'])
        
        # Concatenate all embeddings into a single tensor.
        x = torch.cat((skill_emb, position_emb, education_emb, job_position_emb), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x.squeeze()

# Define the number of unique categories for each feature.
num_skills = df['skills_enc'].nunique()
num_positions = df['positions_enc'].nunique()
num_education = df['education_enc'].nunique()
num_job_position = df['job_position_enc'].nunique()

# Instantiate the model.
model = JobMatchModel(num_skills, num_positions, num_education, num_job_position, embedding_dim=8)

# Define the loss function and optimizer.
criterion = nn.MSELoss()  # Mean Squared Error loss for regression.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Set tolerance for accuracy calculation (if needed).
tolerance = 0.1  # A prediction is considered "accurate" if |prediction - target| < tolerance.

# Training loop for the model.
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    total_samples = 0
    for features, target in train_loader:
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * target.size(0)
        
        # Calculate the number of correct predictions in this batch.
        batch_corrects = ((output - target).abs() < tolerance).sum().item()
        running_corrects += batch_corrects
        total_samples += target.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = running_corrects / total_samples
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

# Evaluate the model on the test set.
model.eval()
test_loss = 0.0
test_corrects = 0.0
total_test_samples = 0
with torch.no_grad():
    for features, target in test_loader:
        output = model(features)
        loss = criterion(output, target)
        test_loss += loss.item() * target.size(0)
        
        batch_corrects = ((output - target).abs() < tolerance).sum().item()
        test_corrects += batch_corrects
        total_test_samples += target.size(0)
test_loss /= len(test_dataset)
test_accuracy = test_corrects / total_test_samples
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Print the predicted output and the actual match score for each user in the test set.
print("\nPredicted vs. Actual Match Scores for each user in the test set:")
with torch.no_grad():
    for features, target in test_loader:
        output = model(features)
        # Convert tensors to lists for printing.
        for pred, actual in zip(output.tolist(), target.tolist()):
            print(f"Predicted: {pred:.4f}, Actual: {actual:.4f}")

# Function to predict the best job for a new user.
def predict_best_job(user_skills, user_positions, user_education):
    # Convert the user's input into indices using the corresponding LabelEncoders.
    skill_idx = torch.tensor([le_skills.transform([user_skills])[0]], dtype=torch.long)
    position_idx = torch.tensor([le_positions.transform([user_positions])[0]], dtype=torch.long)
    education_idx = torch.tensor([le_education.transform([user_education])[0]], dtype=torch.long)

    best_score = 0.0
    best_job = None
    # Loop over all possible job positions (30 options).
    for job in le_job_position.classes_:
        job_idx = torch.tensor([le_job_position.transform([job])[0]], dtype=torch.long)
        features = {
            'skills': skill_idx,
            'positions': position_idx,
            'education': education_idx,
            'job_position': job_idx
        }
        score = model(features).item()
        if score > best_score:
            best_score = score
            best_job = job
    return best_job, best_score

# Example usage of the prediction function.
new_user_skills = "Python, Machine Learning"
new_user_positions = "Data Analyst"
new_user_education = "Bachelor Computer Science"

predicted_job, match_score = predict_best_job(new_user_skills, new_user_positions, new_user_education)
print("\nRecommended Job:", predicted_job, "with match score:", match_score)
