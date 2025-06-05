# Brain Tumor Detection System

An AI-powered web application for detecting brain tumors from MRI scans using deep learning.

## Features

- Upload and analyze MRI scans
- Real-time tumor detection using AI
- Patient information management
- Prediction history tracking
- Modern and responsive UI
- Secure file handling

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **AI/ML**: TensorFlow, Keras
- **Data Processing**: NumPy, Pandas, PIL

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

## Project Structure

```
brain-tumor-detection/
├── app.py              # Main application file
├── requirements.txt    # Project dependencies
├── static/            # Static files (CSS, JS, images)
├── templates/         # HTML templates
└── README.md          # Project documentation
```

## Usage

1. Open your web browser and navigate to `http://localhost:5000`
2. Click "Upload MRI Scan" to start a new analysis
3. Enter patient details and upload the MRI scan
4. View the analysis results
5. Access prediction history from the home page

## Deployment

The application can be deployed on various platforms:

1. **Heroku**:
   - Create a `Procfile` with: `web: gunicorn app:app`
   - Set up environment variables
   - Deploy using Heroku CLI

2. **AWS**:
   - Use Elastic Beanstalk
   - Configure environment variables
   - Set up S3 for file storage

3. **Google Cloud Platform**:
   - Deploy on App Engine
   - Use Cloud Storage for files
   - Set up Cloud SQL if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow and Keras for the ML framework
- Flask for the web framework
- All contributors and supporters
