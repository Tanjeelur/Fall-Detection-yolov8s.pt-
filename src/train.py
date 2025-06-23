from ultralytics import YOLO
import os

def train_model():
    # Set the working directory to the folder containing the script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Verify the working directory
    print(f"Current Working Directory: {os.getcwd()}")

    # Get the absolute path to the .yaml file
    conf_path = os.path.abspath('data.yaml')
    print(f"Absolute path of conf.yaml: {conf_path}")

    # Check if the file exists
    if not os.path.exists(conf_path):
        print("Error: conf.yaml does not exist at the specified path.")
    else:
        print("dataset3.yaml found!")

        # Load a pre-trained YOLOv8 model
        model = YOLO('yolov8n.pt')  # Load a pre-trained model (recommended for fine-tuning)

        # Train the model
        results = model.train(
            data=conf_path,  # Use the verified path
            epochs=150,       # Number of training epochs
            imgsz=640,       # Image size
            batch=8,        # Batch size
            name='yolov11_fall_detection'  # Name of the training run
        )

if __name__ == '__main__':
    train_model()
