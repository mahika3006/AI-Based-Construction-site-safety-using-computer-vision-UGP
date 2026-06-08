# AI-Based Construction Site Safety Monitoring Using Computer Vision (UGP)

An advanced real-time safety compliance and hazard detection system for construction sites using custom-trained **YOLOv8** object detection models and **OpenCV**.

---

## 📌 Project Overview
Construction sites present highly hazardous environments, contributing to a significant portion of global workplace accidents (falls, machinery collisions, lack of safety gear compliance). Traditional manual safety inspection is labor-intensive and lacks 24/7 coverage. 

This project transforms standard, underutilized site CCTV feeds into **smart, automated safety monitoring systems** that detect safety compliance, machinery location, and potential hazards in real time.

---

## 🚀 Key Features & Detections
The system is divided into multiple modular detection pipelines that run concurrently on video streams:
* 👷‍♂️ **Personal Protective Equipment (PPE) Detection**: Real-time identification of workers with safety helmets, vests, and gloves.
* 🏗️ **Heavy Machinery Tracking**: Identification of site machinery including Cranes, Excavators, and Concrete Mixers/Dumpers.
* 🚧 **Structural Safety**: Detection of exposed Reinforcement Bars (rebars) and active excavation zones.
* 🎥 **Multi-Model Inference integration (`all.py`)**: Combines multiple specialty YOLOv8 models to monitor persons, machinery, and safety compliance in a single output stream.

---

## 🛠️ Tech Stack & Workflow
* **Deep Learning Model**: YOLOv8 (Ultralytics)
* **Computer Vision**: OpenCV (Python)
* **Dataset Management**: Annotated and structured via Roboflow
* **Training Platform**: Google Colab (GPU accelerated) for custom training (producing model weights `.pt`)
* **Local Development & Deployment**: VS Code using locally optimized scripts with virtual environment support.

---

## 📂 Project Structure & Scripts
* `UGP.py`: Standard entry script using default YOLOv8n to monitor persons and vehicles.
* `all.py`: Unified pipeline running multiple YOLOv8 models simultaneously for person, crane, mixer, excavator, rebar, and PPE safety detection.
* `safeperson.py`: Focused safety gear compliance checking (helmets, vests, gloves).
* `excavatordetection.py`: Specialized detection script for excavators using Oriented Bounding Boxes (OBB).
* `mixer.py`: Dedicated script for concrete mixers/dumpers tracking.
* `reinbars.py`: Detection script for exposed reinforcement bars (rebars).
* `crane.train.py`: Script used for custom training configuration of crane detection models.
* `data.yaml`: Dataset configuration details containing class names and dataset paths.

---

## 📈 Future Roadmap (Next Semester)
1. **Low-Visibility Optimization**: Enhance detection accuracy in blurry frames, low-light conditions, and low frame rate CCTV streams.
2. **Quantitative Risk Analysis**:
   * Calculate proximity/distance between workers and heavy machinery.
   * Determine safe vs. unsafe zones based on active equipment.
   * Output a real-time **% Safety Score** (e.g., *Safe: 85-100%, Moderate: 60-85%, High Risk: <60%*).
3. **Safety Dashboard**: Build an interactive web dashboard showing real-time statistics, incident alerts, and site compliance percentages.

---

## 🏆 Key Achievements
* **Selected for SEC 2026**: This research has been selected for presentation at the **Structural Engineering Convention (SEC 2026)**, scheduled to be held from **20th-23rd December 2026** at the **Indian Institute of Technology (IIT) Hyderabad**.

---

## 👥 Authors & Guidance
* **Author**: Mahika Chaudhary
* **Project Guide**: Dr. Mahendra Kumar Pal