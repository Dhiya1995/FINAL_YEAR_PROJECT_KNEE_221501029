# Knee Osteoarthritis Classification System

A comprehensive AI-powered system for classifying knee X-ray images into Kellgren-Lawrence (KL) grades with explainable AI, multilingual report generation, and prescriptive care recommendations.

## Features

### 1. Web Interface Module
- User authentication and registration
- Secure login system with SQLite database
- User profile management (height, weight, age, activity level)
- Image upload interface
- Language selection for reports
- Interactive results dashboard
- Downloadable PDF reports

### 2. Deep Learning Classification Module
- ResNet50 or DenseNet121 based CNN classifier
- KL grade classification (0-4)
- MC Dropout for uncertainty estimation
- Grad-CAM visualization for model explainability
- Comprehensive metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix generation

### 3. GAN Synthesis Module
- StyleGAN2-ADA integration for synthetic image generation
- Dataset balancing for minority classes
- FID (Frechet Inception Distance) validation
- Automatic dataset augmentation

### 4. Radiology Report Generator Module
- Med-Gemma LLM integration for report generation
- Google Translate API for multilingual support
- Structured clinical reports
- PDF generation with Grad-CAM visualization

### 5. Prescriptive Report Module
- Gemini API integration for personalized care plans
- Patient-specific recommendations
- Lifestyle, exercise, and treatment guidance
- PDF export functionality

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
cd knee
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Windows PowerShell
$env:SECRET_KEY="your-secret-key-here"
$env:GOOGLE_TRANSLATE_API_KEY="your-google-translate-api-key"
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:CUDA_AVAILABLE="true"  # if using GPU

# Linux/Mac
export SECRET_KEY="your-secret-key-here"
export GOOGLE_TRANSLATE_API_KEY="your-google-translate-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export CUDA_AVAILABLE="true"
```

4. Initialize the database:
```bash
python -c "from database import init_db; init_db()"
```

## Usage

### Training the Model

1. Train the classification model:
```bash
python models/classification_model.py
```

The model will be saved to `models/best_kl_classifier.pth` after training.

### Running the Web Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Register a new account or login

4. Upload a knee X-ray image

5. View results and generate reports

### Dataset Structure

The dataset should be organized as follows:
```
new-dataset/
├── train/
│   ├── 0/  (KL Grade 0 images)
│   ├── 1/  (KL Grade 1 images)
│   ├── 2/  (KL Grade 2 images)
│   ├── 3/  (KL Grade 3 images)
│   └── 4/  (KL Grade 4 images)
├── val/
│   └── (same structure as train)
└── test/
    └── (same structure as train)
```

## Configuration

Edit `config.py` to customize:
- Model architecture (ResNet50 or DenseNet121)
- Image size
- Batch size
- Learning rate
- Number of epochs
- Supported languages

## API Keys Setup

### Google Translate API
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Cloud Translation API
3. Create credentials and get API key
4. Set `GOOGLE_TRANSLATE_API_KEY` environment variable

### Gemini API
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Set `GEMINI_API_KEY` environment variable

### Med-Gemma Model
1. The system uses Hugging Face transformers to load Med-Gemma
2. Model path can be configured in `config.py`
3. First run will download the model automatically

## Project Structure

```
knee/
├── app.py                      # Flask web application
├── config.py                   # Configuration settings
├── database.py                 # Database operations
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/
│   ├── classification_model.py # CNN classifier
│   ├── gradcam.py             # Grad-CAM implementation
│   └── gan_synthesis.py       # GAN synthesis module
├── reports/
│   ├── radiology_report.py    # Report generator
│   ├── prescriptive_report.py # Care plan generator
│   └── pdf_generator.py       # PDF creation
├── templates/
│   ├── base.html              # Base template
│   ├── login.html             # Login page
│   ├── register.html          # Registration page
│   ├── dashboard.html         # User dashboard
│   ├── profile.html           # Profile settings
│   ├── upload.html            # Image upload
│   └── results.html           # Results display
├── uploads/                    # Uploaded images
├── outputs/                    # Generated reports and visualizations
├── models/                     # Saved model weights
└── new-dataset/                # Training dataset
```

## Metrics and Formulas

### Classification Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Loss Function
- **Cross Entropy Loss**: -Σ yi log(pi)
  - yi = true label (one-hot encoded)
  - pi = predicted probability for class i

### FID Score
- **FID**: ||μr - μs||² + Tr(Σr + Σs - 2(Σr Σs)^(1/2))
  - Lower FID indicates higher similarity between real and synthetic distributions

## Troubleshooting

### Model Not Found Error
- Train the model first using `python models/classification_model.py`
- Ensure the model is saved to `models/best_kl_classifier.pth`

### CUDA Out of Memory
- Reduce batch size in `config.py`
- Use CPU mode by setting `CUDA_AVAILABLE=false`

### API Key Errors
- Ensure all API keys are set as environment variables
- Check API key validity and quotas

## License

This project is for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on the repository.


