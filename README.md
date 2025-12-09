fatigue detection using facial landmarks
to run the code 
cmd - powershell
.venv/scripts/activate - virtual environment 
cd fatigue
streamlit run fatigue_dashboard1.py -main.py
for running second code for headtilt 
streamlit run fatigue_dashboard2.py
+ if headtilt detection is needed refer fatigue_detection2.py
+ due to github limitation i am unable to upload large files refer the full file in
  google drive link for reference -> https://drive.google.com/drive/folders/11ukskfMMFfB2eUq9MschmeGzSRiPH8Py?usp=drive_link

there is an isssue in requirements.txt if in case of running run the requirments.txt outside the file only for docker purpose 

Computer Vision: OpenCV, MediaPipe FaceMesh
ML Models: RandomForest, CNN (TensorFlow), Joblib (.pkl/.h5)
UI: Streamlit (Multi-page dashboard)
Audio: Playsound/Pygame (Non-blocking alarms)
Data: Pandas (Event logging & trends)
Deployment: Docker

text

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- Preferred IDE: **PyCharm** or **VS Code**

### 1. Clone & Setup
git clone <your-repo>
cd fatigue_detection

text

### 2. Virtual Environment
Windows
.venv\Scripts\activate

Linux/Mac
source .venv/bin/activate

text

### 3. Install Dependencies
pip install -r requirements.txt

text

### 4. Run Dashboard
cd fatigue
cd fatigueapi
streamlit run fatigue_dashboard1.py


for optional code(with head tilt)
same folder use the same step only change 
streamlit run fatigue_dashboard2.py



### 5. Run As Docker 
docker build -t fatigue-detection .
This creates an image named fatigue-detection from your Dockerfile in the current folder.

Run command
After a successful build, run the container:

bash
docker run -p 8501:8501 fatigue-detection
Then open:

text
http://localhost:8501

