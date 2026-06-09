from data_preprocessing import DataPreprocessor
from model_training import UNetTrainer
from post_processing import SegmentationPostProcessor


IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"


preprocessor = DataPreprocessor(
    IMAGE_DIR,
    MASK_DIR
)

X_train, X_val, y_train, y_val = (
    preprocessor.split_data()
)

trainer = UNetTrainer()

model = trainer.build_model()

history = trainer.train(
    model,
    X_train,
    y_train,
    X_val,
    y_val
)

processor = SegmentationPostProcessor()

prediction, inference_time = (
    processor.predict(
        X_val[0],
        model
    )
)

print(
    f"Inference Time: "
    f"{inference_time:.4f} seconds"
)

processor.visualize(
    X_val[0],
    prediction,
    y_val[0]
)

model.save(
    "models/final_UNET_Butterfly_segmentation.keras"
)