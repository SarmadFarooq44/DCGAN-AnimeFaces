# DCGAN-AnimeFaces

This repository contains code for training a DCGAN on the Anime Face dataset. It includes:
- Training the GAN using Binary Cross-Entropy Loss and Adam optimizer.
- Saving generated images every 10 epochs.
- An endpoint for latent space interpolation.
- A Flask API to generate images on demand.

## Repository Structure
- **GAN.py:** Code to train the DCGAN.
- **app.py:** Flask application to serve image generation endpoints.
- **requirements.txt:** Python package dependencies.

## Setup Instructions
1. **Clone the Repository:**
   git clone https://github.com/yourusername/DCGAN-AnimeFaces.git
   cd DCGAN-AnimeFaces

2. **Install Dependencies:**
   pip install -r requirements.txt

3. **Download the Dataset:**
   The code and link to download dataset is included in GAN.py.

2. **Train the Model:**
   python GAN.py

2. **Run the Flask Application:**
   python app.py
   This will start the server on http://0.0.0.0:5000.
