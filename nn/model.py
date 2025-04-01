import pandas as pd                # עבודה עם קבצי CSV
import numpy as np                 # לעבודה עם מערכים ומתמטיקה
import torch                       # PyTorch
import torch.nn as nn              # מודולי רשתות נוירונים
import torch.optim as optim        # אופטימייזרים
from torch.utils.data import Dataset, DataLoader  # ניהול נתונים
from collections import Counter    # ספירת מילים לבניית מילון
import re                          # לעבודה עם ביטויים רגולריים

# פונקציה לטוקניזציה – הסרת תווים מיוחדים, המרת טקסט לאותיות קטנות ופיצול למילים
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return text.split()

# טעינת הנתונים מקובץ CSV (יש לשנות את שם הקובץ בהתאם)
data = pd.read_csv("/content/resume_data.csv")

# אנו מניחים שיש לנו את העמודות:
# skills, degree_names, major_field_of_studies, positions, job_position_name, matched_score

# הגדרת עמודת education כמשלוב של degree_names ו־major_field_of_studies
data['education'] = data['degree_names'].astype(str) + " " + data['major_field_of_studies'].astype(str)

# בניית מילון (Vocabulary) משותף לכל המאפיינים:
# נעבור על העמודות: skills, education, positions
all_text = []
for col in ['skills', 'education', 'positions']:
    for text in data[col]:
        all_text.extend(tokenize(str(text)))

word_counts = Counter(all_text)
# נבנה מילון שבו כל מילה מקבלת אינדקס (מתחילים מ-1, כאשר 0 שמור לפדינג)
word2idx = {word: idx+1 for idx, (word, count) in enumerate(word_counts.most_common())}
word2idx["<PAD>"] = 0
vocab_size = len(word2idx)
print("גודל המילון:", vocab_size)

# המרת התוויות (job_position_name) למספרים
labels = data['job_position_name'].unique()
label2idx = {label: idx for idx, label in enumerate(labels)}
num_classes = len(label2idx)
print("מספר קטגוריות:", num_classes)

# הגדרת מחלקת Dataset אשר מחזירה 3 רצפים (אחד לכל מאפיין) ואת התווית
class JobDatasetSeparate(Dataset):
    def __init__(self, df, word2idx, label2idx, max_len_skills=50, max_len_education=30, max_len_positions=50):
        """
        df: DataFrame המכיל את הנתונים.
        word2idx: מילון המרה של מילים לאינדקסים.
        label2idx: מילון המרה של תוויות למספרים.
        max_len_*: אורך מקסימלי לכל מאפיין.
        """
        self.df = df
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.max_len_skills = max_len_skills
        self.max_len_education = max_len_education
        self.max_len_positions = max_len_positions

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # שליפת המאפיינים הנפרדים
        text_skills = str(row['skills'])
        text_education = str(row['education'])
        text_positions = str(row['positions'])

        # טוקניזציה לכל מאפיין
        tokens_skills = tokenize(text_skills)
        tokens_education = tokenize(text_education)
        tokens_positions = tokenize(text_positions)

        # המרה לאינדקסים באמצעות המילון, כאשר אם מילה לא נמצאה – מקבלים 0 (PAD)
        seq_skills = [self.word2idx.get(token, 0) for token in tokens_skills]
        seq_education = [self.word2idx.get(token, 0) for token in tokens_education]
        seq_positions = [self.word2idx.get(token, 0) for token in tokens_positions]

        # התאמה לאורך קבוע לכל מאפיין – חיתוך או הוספת פדינג
        if len(seq_skills) < self.max_len_skills:
            seq_skills = seq_skills + [0]*(self.max_len_skills - len(seq_skills))
        else:
            seq_skills = seq_skills[:self.max_len_skills]

        if len(seq_education) < self.max_len_education:
            seq_education = seq_education + [0]*(self.max_len_education - len(seq_education))
        else:
            seq_education = seq_education[:self.max_len_education]

        if len(seq_positions) < self.max_len_positions:
            seq_positions = seq_positions + [0]*(self.max_len_positions - len(seq_positions))
        else:
            seq_positions = seq_positions[:self.max_len_positions]

        # המרה לטנסורים
        seq_skills = torch.tensor(seq_skills, dtype=torch.long)
        seq_education = torch.tensor(seq_education, dtype=torch.long)
        seq_positions = torch.tensor(seq_positions, dtype=torch.long)

        # המרת תווית המשרה למספר
        label = self.label2idx[row['job_position_name']]
        label = torch.tensor(label, dtype=torch.long)

        return seq_skills, seq_education, seq_positions, label

# חלוקת הנתונים לאימון ובדיקה (80%-20%)
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = JobDatasetSeparate(train_df, word2idx, label2idx, max_len_skills=50, max_len_education=30, max_len_positions=50)
test_dataset = JobDatasetSeparate(test_df, word2idx, label2idx, max_len_skills=50, max_len_education=30, max_len_positions=50)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# הגדרת מודל עם 3 סניפים – אחד לכל מאפיין
class MultiFeatureGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=1, dropout=0.5):
        """
        vocab_size: גודל המילון.
        embed_size: מימד ה-embedding לכל מילה.
        hidden_size: מספר היחידות בשכבת GRU בכל סניף.
        output_size: מספר קטגוריות הסיווג.
        num_layers: מספר שכבות GRU בכל סניף.
        dropout: שיעור דראופאוט למניעת overfitting.
        """
        super(MultiFeatureGRUClassifier, self).__init__()
        # שכבות embedding נפרדות לכל מאפיין
        self.embedding_skills = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embedding_education = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embedding_positions = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # סניפי GRU נפרדים לכל מאפיין
        self.gru_skills = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.gru_education = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0)
        self.gru_positions = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0)
        # שכבה Fully Connected הממפה את האיחוד של כל הסניפים לקטגוריות הסופיות
        self.fc = nn.Linear(3 * hidden_size, output_size)

    def forward(self, seq_skills, seq_education, seq_positions):
        # עיבוד מאפיין ה-skills
        embedded_skills = self.embedding_skills(seq_skills)   # גודל: [batch, seq_len, embed_size]
        _, h_skills = self.gru_skills(embedded_skills)          # h_skills: [num_layers, batch, hidden_size]
        h_skills_last = h_skills[-1]                            # בחירת השכבה האחרונה

        # עיבוד מאפיין ה-education
        embedded_education = self.embedding_education(seq_education)
        _, h_education = self.gru_education(embedded_education)
        h_education_last = h_education[-1]

        # עיבוד מאפיין ה-positions
        embedded_positions = self.embedding_positions(seq_positions)
        _, h_positions = self.gru_positions(embedded_positions)
        h_positions_last = h_positions[-1]

        # איחוד (concatenation) של הייצוגים מכל הסניפים
        combined = torch.cat((h_skills_last, h_education_last, h_positions_last), dim=1)
        output = self.fc(combined)
        return output

# הגדרת היפר-פרמטרים למודל
embed_size = 128         # מימד ה-embedding
hidden_size = 256        # מספר היחידות ב-GRU בכל סניף
num_layers = 1           # מספר שכבות GRU
dropout = 0.2           # שיעור הדראופאוט
learning_rate = 0.001    # שיעור הלמידה
num_epochs = 32          # מספר epochs לאימון

model = MultiFeatureGRUClassifier(vocab_size=vocab_size, embed_size=embed_size, hidden_size=hidden_size,
                                  output_size=num_classes, num_layers=num_layers, dropout=dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# הגדרת פונקציית האובדן והאופטימייזר
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# לולאת אימון המודל
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for seq_skills, seq_education, seq_positions, labels in train_loader:
        seq_skills = seq_skills.to(device)
        seq_education = seq_education.to(device)
        seq_positions = seq_positions.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(seq_skills, seq_education, seq_positions)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"אפך {epoch+1}/{num_epochs}, אובדן ממוצע: {avg_loss:.4f}")

    # הערכת ביצועים על סט הבדיקה
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seq_skills, seq_education, seq_positions, labels in test_loader:
            seq_skills = seq_skills.to(device)
            seq_education = seq_education.to(device)
            seq_positions = seq_positions.to(device)
            labels = labels.to(device)

            outputs = model(seq_skills, seq_education, seq_positions)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"דיוק על סט בדיקה: {accuracy:.4f}")

# שמירת המודל המאומן
torch.save(model.state_dict(), "multi_feature_gru_classifier.pth")
