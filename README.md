# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
1. Develop a binary classification model using a pretrained VGG19 to distinguish between defected and non-defected capacitors by modifying the last layer to a single neuron.  
2. Train the model on a dataset containing images of various defected and non-defected capacitors to improve defect detection accuracy.  
3. Optimize and evaluate the model to ensure reliable classification for capacitor quality assessment in manufacturing.

## DESIGN STEPS
### STEP 1:
Collect and preprocess the dataset containing images of defected and non-defected capacitors.

### STEP 2:
Split the dataset into training, validation, and test sets.

### STEP 3:
Load the pretrained VGG19 model with weights from ImageNet.

### STEP 4:
Remove the original fully connected (FC) layers and replace the last layer with a single neuron (1 output) with a Sigmoid activation function for binary classification.

### STEP 5:
Train the model using binary cross-entropy loss function and Adam optimizer.

### STEP 6:
Evaluate the model with test data loader and intepret the evaluation metrics such as confusion matrix and classification report.

## PROGRAM

```python
# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)

for param in model.parameters():
  param.requires_grad = False


# Modify the final fully connected layer to match one binary classes
num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features,1)


# Include the Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)


# Train the model
def train_model(model, train_loader, test_loader, epochs=10):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        # -------- TRAIN --------
        model.train()
        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        val_loss = val_loss / len(test_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print("Name:MOHAMED RASHITH S ")
    print("Register Number: 212223243003     ")
    # Plot Loss
    plt.plot(train_losses,label="Train Loss")
    plt.plot(val_losses,label="Validation Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot

<img width="477" height="578" alt="image" src="https://github.com/user-attachments/assets/d1f9d6c0-cb5e-4f0b-94be-9552db532d55" />

### Classification Report

<img width="477" height="205" alt="image" src="https://github.com/user-attachments/assets/a5d4948d-fa89-4adf-869b-d0119eab328c" />


### Confusion Matrix


<img width="486" height="473" alt="image" src="https://github.com/user-attachments/assets/cc3ec9e8-ffa2-4096-ba2a-8fa4fddb103e" />


### New Sample Prediction
<img width="492" height="385" alt="image" src="https://github.com/user-attachments/assets/b6ed9624-c85a-47d3-85a7-7c6489f80601" />

<img width="448" height="428" alt="image" src="https://github.com/user-attachments/assets/c7c5c0a1-1ea1-4968-a626-8c2a4b8b43d3" />




## RESULT
The VGG-19 model was successfully trained and optimized to classify defected and non-defected capacitors.
