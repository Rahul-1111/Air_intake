conda create -n yolo_env python=3.11 -y
conda activate yolo_env
pip install ultralytics pymcprotocol opencv-python

conda env remove -n yolo_venv


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

