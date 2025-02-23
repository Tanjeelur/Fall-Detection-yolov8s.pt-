from ultralytics import YOLO
import os

def train_model():
    # Set the working directory to the folder containing the script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Verify the working directory
    print(f"Current Working Directory: {os.getcwd()}")

    # Get the absolute path to the .yaml file
    conf_path = os.path.abspath('data.yaml')
    print(f"Absolute path of data.yaml: {conf_path}")

    # Check if the file exists
    if not os.path.exists(conf_path):
        print("Error: data.yaml does not exist at the specified path.")
    else:
        print("data.yaml found!")

        # Load the YOLOv8n model
        model = YOLO('yolov8n.pt')  # Load the pre-trained YOLOv8n model

        # Train the model
        results = model.train(
            data=conf_path,  # Path to the dataset configuration file
            epochs=200,      # Number of training epochs
            imgsz=640,       # Image size (adjust based on GPU memory)
            batch=8,         # Batch size (reduce if out of memory)
            name='yolov8n_fall_detection',  # Name of the training run
            augment=True,    # Enable data augmentation
            hsv_h=0.015,     # Adjust hue
            hsv_s=0.7,       # Adjust saturation
            hsv_v=0.4,       # Adjust value (brightness)
            translate=0.1,   # Randomly translate images
            scale=0.5,       # Randomly scale images
            flipud=0.5,      # Flip images vertically
            fliplr=0.5,      # Flip images horizontally
            mosaic=1.0,      # Enable mosaic augmentation
            mixup=0.1,       # Enable mixup augmentation
            weight_decay=0.0005,  # L2 regularization (weight decay)
            patience=20,     # Early stopping (stop if no improvement for 20 epochs)
            lr0=0.001,       # Initial learning rate
            lrf=0.01,        # Final learning rate
            workers=4,       # Use 4 CPU cores for data loading
        )

if __name__ == '__main__':
    train_model()