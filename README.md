## Getting Started

### Cloning the Repository

~~~bash
git clone https://github.com/yomi-adamo/ai-recognition-system.git
cd ai-recognition-system.git
~~~

### Adding Images for Recognition
- Add .jpg images to the reference-faces directory for the ai to train itself on

### Setup Environment
~~~bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Mac/Linux

pip install face_recognition opencv-python requests numpy
~~~

### Folders
The logs folder will be automatically generated when you run the code

### Run the Program
pyton demo(demo#).py
