# Stanford-Dogs-Dataset-Classification

## Project Overview
This project involves developing a deep learning model to categorize images from the Stanford Dogs dataset using TensorFlow and Keras. The project adapts and optimizes a convolutional neural network (CNN) architecture to accurately identify dog breeds among 120 different classes.

## Dataset
The Stanford Dogs dataset contains 20,580 images, which are divided into 12,000 training images and 8,580 testing images, representing 120 breeds of dogs from around the world. Each image is annotated with a breed label and bounding box information. This dataset, sourced from ImageNet, is primarily used for fine-grained image categorization tasks.

## Model Architecture
The model utilizes a pre-trained ResNet50 architecture as a foundational feature extractor, with the top layer replaced by a Global Average Pooling 2D layer followed by a dense output layer with softmax activation to classify dog breeds. The model benefits from transfer learning, utilizing weights from ImageNet.

## Data Preprocessing
Images are resized to 224x224 pixels to conform to the input requirements of ResNet50. The data is split into training, validation, and testing sets to ensure comprehensive training and effective evaluation.

## Training
Training steps include setting a random seed for reproducibility, preprocessing images using ResNet50's input function, and defining callbacks for early stopping and learning rate adjustments to optimize training phases.

### Training Configurations
- **Callbacks**: Implementation of early stopping, learning rate adjustments, and model checkpointing to enhance training effectiveness.
- **Optimizer**: Use of the Adam optimizer with a learning rate of 0.01.

## Testing and Performance
The model's performance is evaluated on the test set to ensure effective generalization to new data. The model achieved an accuracy of 74.32% on the test dataset with a loss of 3.4431, demonstrating its capability to generalize well to unseen data.

## Results
The project includes detailed accuracy metrics and loss graphs plotted using Matplotlib to visualize the model's learning progression over time.

## Future Work
Potential enhancements could include:
- Employing more advanced techniques such as data augmentation.
- Exploring different CNN architectures.
- Fine-tuning hyperparameters to further improve accuracy and model robustness.
