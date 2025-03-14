import os
from flask import Flask, send_file, render_template_string
import torch
import torch.nn as nn
import torchvision.utils as vutils

app = Flask(__name__)

# Ensure static directory exists for saving images
STATIC_DIR = 'static'
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Device and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
image_channels = 3
hidden_dim = 64

# Generator Network
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Load the trained generator
netG = Generator().to(device)
netG.load_state_dict(torch.load('generator.pth', map_location=device))
netG.eval()

# Define spherical linear interpolation (slerp)
def slerp(val, low, high):
    """Spherical interpolation between two vectors."""
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    dot = torch.clamp(torch.dot(low_norm, high_norm), -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high

# Main page with buttons to run functions
@app.route('/')
def index():
    html = """
    <html>
      <head>
        <title>DCGAN API Main Page</title>
        <style>
          body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; }
          button { padding: 15px 25px; font-size: 16px; margin: 10px; cursor: pointer; }
        </style>
      </head>
      <body>
        <h1>Welcome to the DCGAN API</h1>
        <p>Click a button below to generate images:</p>
        <button onclick="window.location.href='/generate'">Generate Single Image</button>
        <button onclick="window.location.href='/interpolate'">Interpolate Images</button>
      </body>
    </html>
    """
    return render_template_string(html)

# Optional: Define a favicon route to handle favicon requests
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/generate')
def generate_image():
    """Generate a single image from random noise and display with buttons."""
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        fake_image = netG(noise).cpu()
        file_path = os.path.join(STATIC_DIR, 'generated_image.png')
        vutils.save_image(fake_image, file_path, normalize=True)
    
    html = f"""
    <html>
      <head>
        <title>Generated Image</title>
        <style>
          body {{ text-align: center; font-family: Arial, sans-serif; margin-top: 20px; }}
          button {{ padding: 10px 20px; font-size: 16px; margin: 10px; }}
          img {{ max-width: 80%; height: auto; }}
        </style>
      </head>
      <body>
        <h1>Generated Image</h1>
        <img src="/static/generated_image.png" alt="Generated Image"><br>
        <button onclick="window.location.href='/generate'">Generate Again</button>
        <button onclick="window.location.href='/'">Go Back</button>
      </body>
    </html>
    """
    return render_template_string(html)

@app.route('/interpolate')
def interpolate_images():
    """Generate smooth interpolation between two latent vectors and display with buttons."""
    steps = 10  # Number of interpolation steps
    with torch.no_grad():
        z_start = torch.randn(1, latent_dim, 1, 1, device=device)
        z_end = torch.randn(1, latent_dim, 1, 1, device=device)
        interpolated_images = []
        
        for i in range(steps):
            alpha = i / (steps - 1)
            z_interp = slerp(alpha, z_start.squeeze(), z_end.squeeze()).view(1, latent_dim, 1, 1)
            fake_img = netG(z_interp)
            interpolated_images.append(fake_img)
        
        interpolated_images = torch.cat(interpolated_images, dim=0).cpu()
        grid = vutils.make_grid(interpolated_images, nrow=steps, normalize=True)
        file_path = os.path.join(STATIC_DIR, 'interpolated_image.png')
        vutils.save_image(grid, file_path, normalize=True)
    
    html = f"""
    <html>
      <head>
        <title>Interpolated Images</title>
        <style>
          body {{ text-align: center; font-family: Arial, sans-serif; margin-top: 20px; }}
          button {{ padding: 10px 20px; font-size: 16px; margin: 10px; }}
          img {{ max-width: 80%; height: auto; }}
        </style>
      </head>
      <body>
        <h1>Interpolated Images</h1>
        <img src="/static/interpolated_image.png" alt="Interpolated Images"><br>
        <button onclick="window.location.href='/interpolate'">Generate Again</button>
        <button onclick="window.location.href='/'">Go Back</button>
      </body>
    </html>
    """
    return render_template_string(html)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
