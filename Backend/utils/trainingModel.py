
import json
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

with open("./datasets/Marketing.json", "r") as file:
    data = json.load(file)

questions, answers = [], []
for item in data:
    questions.append(item["question"])
    answers.append(item["answer"])


tokenizer_question = Tokenizer()
tokenizer_question.fit_on_texts(questions)

question_encoded = tokenizer_question.texts_to_sequences(questions)
question_padded = pad_sequences(question_encoded, padding="post")


encoder = LabelEncoder()
answer_encoded = encoder.fit_transform(answers)
num_classes = len(set(answer_encoded)) 


X = mx.array(np.array(question_padded), dtype=mx.float32)
y = mx.array(np.array(answer_encoded), dtype=mx.int32)


X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.1, random_state=42)

X_train = mx.array(X_train, dtype=mx.float32)
X_test = mx.array(X_test, dtype=mx.float32)
y_train = mx.array(y_train, dtype=mx.int32)
y_test = mx.array(y_test, dtype=mx.int32)


def one_hot_encoding(labels, num_classes):
    return mx.equal(mx.expand_dims(labels, -1), mx.arange(num_classes))


y_train = one_hot_encoding(y_train, num_classes).astype(mx.float32)
y_test = one_hot_encoding(y_test, num_classes).astype(mx.float32)


class MarketingModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, output_dim)
        self.activation = nn.ReLU()

    def __call__(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


input_dim = X_train.shape[1]
model = MarketingModel(input_dim, num_classes)


loss = nn.losses.cross_entropy

def loss_fn(model):
    logits = model(X_train)
    return loss(logits, y_train).mean()

epochs_num = 2500
optimizer = optim.SGD(learning_rate=0.001,weight_decay=0.01)

for epoch in range(epochs_num):
    def compute_loss(model): 
        return loss_fn(model)
    loss_value, grads = mx.value_and_grad(compute_loss, argnums=0)(model)
    optimizer.update(model, grads)
    print(f"Epoch [{epoch+1}/{epochs_num}], Loss: {loss_value.item():.4f}" )

print(mx.default_device())
def input_question(question):
    question_seq = tokenizer_question.texts_to_sequences([question])
    question_padded = pad_sequences(question_seq, maxlen = X_train.shape[1], padding="post")
    question_tensor = mx.array(np.array(question_padded), dtype=mx.float32)
    prediction = model(question_tensor)
    predicted_index = prediction.argmax(axis=1).item()
    confidence = mx.softmax(prediction).max().item()  
    if confidence < 0.5:
        return "Sorry, I don't understand your question."
    else:
        return answers[predicted_index]









