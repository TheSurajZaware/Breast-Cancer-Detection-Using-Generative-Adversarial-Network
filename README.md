# Breast Cancer Detection Using a DCGAN and Transfer Learning

## 📌 Description

This project explores **breast ultrasound–based tumor screening** by combining:

- A **Deep Convolutional GAN (DCGAN)** built with **TensorFlow/Keras** to synthesize additional **benign**-class images and enrich the training distribution.
- A **binary classifier** based on **ResNet-152** with a custom head, trained in **PyTorch** via **transfer learning** (benign vs. malignant).
- A **Flask** web app that serves a simple UI and exposes a **`/predict`** API for inference on uploaded images.

The GAN is used for **data augmentation / synthetic sample generation**, not as the final discriminator for cancer class. Classification is handled by the ResNet model.

---

## 🎯 Motivation / Problem Statement

Medical imaging datasets are often **small and class-imbalanced**. Training only on limited real samples can hurt generalization. Generating plausible extra **benign** ultrasound–like images with a DCGAN can **expand the benign manifold** before (or alongside) supervised training. A pretrained **ResNet** backbone then learns a strong feature representation faster than training a CNN from scratch, which supports a practical **benign vs. malignant** decision tool and a demo web interface.

---

## ✨ Features

- 📊 **DCGAN pipeline** (notebook): load/resize RGB images, optional `.npy` cache, generator + discriminator training, preview grids, export of many synthetic PNGs.
- 🧠 **ResNet-152 transfer learning** (notebook): `ImageFolder` train/valid/test splits, augmentation on train, SGD + cross-entropy, saving checkpoints (`.pkl` / `.model`).
- 🌐 **Web demo**: static landing page (awareness content + team), image upload, **POST `/predict`** with base64 image payload, JSON response with predicted label.
- 📁 **Sample outputs**: folder for DCGAN-generated images with notes on quality vs. training length.

---

## 🛠 Tech Stack

| Area | Technologies |
|------|----------------|
| DCGAN | TensorFlow 2.x, Keras, NumPy, PIL, Matplotlib |
| Classification | PyTorch, `torchvision` (ResNet-152, transforms) |
| Web | Flask, jQuery, Bootstrap (Medilab-style template) |
| Imaging | OpenCV (`cv2`), Pillow |

> **Note:** `Web_App/app.py` imports TensorFlow but inference uses the **PyTorch** `model.pkl`. TensorFlow is required if you run the DCGAN notebook locally.

---

## 📂 Project Structure

> High-level layout (vendor assets under `Web_App/static/` are omitted for clarity).

```
Breast-Cancer-Detection-Using-Generative-Adversarial-Network/
├── DCGAN_Model/
│   └── BCD_DCGAN.ipynb          # DCGAN training & synthetic image export
├── DCGAN_Generated_Samples/     # Placeholder / notes for generated images
├── Transfer_Learning_Model_ResNet/
│   └── BCD_TL_ResNet.ipynb      # ResNet-152 training (Colab-oriented paths)
├── Transfer_Learning_Validation_Model/
│   └── BCD_TL_Validation.ipynb  # Load model & sanity-check inference
├── Web_App/
│   ├── app.py                   # Flask API + PyTorch inference
│   ├── model.pkl                # Trained weights (obtain separately; see below)
│   ├── templates/
│   │   └── index.html
│   └── static/                  # CSS, JS, images, third-party libraries
└── README.md
```

---

## ⚙️ How It Works

1. **Data (real ultrasound)**  
   Organize images for the DCGAN (e.g. benign-focused folder as in the notebook) and for classification using **`torchvision.datasets.ImageFolder`** with `train/`, `valid/`, and `test/` subfolders and class-named directories inside each.

2. **DCGAN (`BCD_DCGAN.ipynb`)**  
   - Resize to **96×96** RGB, scale pixels to **[-1, 1]** (to match `tanh` output).  
   - Optional **`.npy`** cache to avoid repeated preprocessing.  
   - **Generator:** dense → reshape → transposed convolutions → **96×96×3** `tanh`.  
   - **Discriminator:** conv blocks + **sigmoid** “real vs. fake.”  
   - Train with alternating generator/discriminator steps; periodically save preview grids and **`face_generator-*.h5`**.  
   - Optionally load a saved generator and write hundreds of **PNG** samples for augmentation.

3. **Classifier (`BCD_TL_ResNet.ipynb`)**  
   - **ResNet-152** trunk + **2-class** linear head.  
   - ImageNet-style **normalize** and **224×224** crops (resize + center crop for val/test).  
   - Train with **SGD**, evaluate on validation and test loaders; save model files for deployment.

4. **Web app (`Web_App/app.py`)**  
   - Load **`model.pkl`** (full PyTorch module).  
   - Decode **base64** image from JSON, apply **resize → center crop → tensor → ImageNet normalize**, run **`model.eval()`**, return **Benign** vs. **Malignant** label.

> **Assumption:** Synthetic DCGAN images are intended to be mixed into (or used to augment) the benign training set; the repository does not ship a single automated script that merges DCGAN outputs into the ResNet dataloader—notebooks are the source of truth.

---

## 📥 Installation

**Prerequisites:** Python 3.8+ recommended, pip, optional CUDA for GPU training.

There is **no** `requirements.txt` in this repo; install dependencies inferred from the code, for example:

```bash
cd Web_App
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install flask torch torchvision pillow numpy opencv-python
# For DCGAN notebooks only:
pip install tensorflow tqdm matplotlib
```

**Trained classifier weights:** download the file referenced in the original project notes and place it as:

`Web_App/model.pkl`

Example (Google Drive link from project documentation—verify availability):

`https://drive.google.com/uc?export=download&id=1IZdaEOVN-Np2SXvY5qW-gabb8dL6KGUC`

---

## 🚀 Usage

### Run the web app (inference)

```bash
cd Web_App
python app.py
```

By default Flask runs with `debug=True`. Open the served URL in a browser, upload an image, and use **Predict**. The frontend posts JSON to `/predict` (see `static/assets/js/main.js`).

> **Assumption / caveat:** `main.js` may reference a **hardcoded** host (`http://127.0.0.1:5000/predict`) and fields like `accuracy` that the current API **does not** return. For portable deployment, use a **relative** URL (e.g. `/predict`) and align the JSON schema between Flask and JavaScript.

### Train DCGAN or ResNet

Open and run cells in order inside:

- `DCGAN_Model/BCD_DCGAN.ipynb`
- `Transfer_Learning_Model_ResNet/BCD_TL_ResNet.ipynb`

Colab-specific cells (Google Drive mount, absolute paths) must be adapted to your machine or cloud paths.

---

## 📀 Dataset Details

- **Source (documented in project):** [BUSI with GT on Kaggle](https://www.kaggle.com/datasets/anaselmasry/datasetbusiwithgt) — breast ultrasound images with associated masks/labels (exact preprocessing splits are defined in your notebooks/local folders).  
- **DCGAN notebook** example: trains on images under a configurable **`DATA_PATH`** (e.g. benign-focused raw images).  
- **Classifier:** expects **`ImageFolder`** layout with **train / valid / test** splits.  
- **Assumptions:** Class indices follow **PyTorch `ImageFolder`** alphabetical ordering; the web app maps **class index 0 → Benign**, **1 → Malignant** in the live prediction branch—ensure your trained model uses the **same** ordering.

---

## 🧬 Model Architecture (Simple Overview)

- **DCGAN**  
  - **Generator** maps a **100-D** random vector through dense and **transposed convolution** layers to a **96×96 RGB** image.  
  - **Discriminator** is a **convolutional** network that outputs a single probability: real vs. generated.  
  - The two networks are trained in opposition: the generator tries to fool the discriminator; the discriminator tries to detect fakes.

- **Classifier**  
  - **ResNet-152** extracts visual features; the original classification layer is replaced by a **2-output** fully connected layer for **benign vs. malignant**.

---

## 📈 Results / Metrics

- Example **training/validation accuracy** trajectories appear in **`BCD_TL_ResNet.ipynb`** outputs (values depend on run, data split, and epochs).  
- **`DCGAN_Generated_Samples/Readme.txt`** notes that quality at **~200 epochs** was **low** and suggests **1000+** epochs for better samples.  
- **Consolidated benchmark table (single number for the whole project):** *to be added* — not centralized in the repository.

---

## 🖼 Screenshots / Outputs

- **Web UI:** landing page, education sections, and **Predict Report** upload area under `Web_App/templates/`.  
- **DCGAN:** preview PNG grids and synthetic images are produced by the notebook (paths are machine-specific).  
- **Repository:** no dedicated `screenshots/` folder; capture your own after running the app.

---

## 🔮 Future Improvements

- Add **`requirements.txt`** / **`environment.yml`** with pinned versions.  
- Fix **API/frontend contract** (relative `/predict`, optional confidence score from softmax).  
- Remove **dead code** at the bottom of `app.py` and unused **TensorFlow** import if only PyTorch is used for serving.  
- Unify **spelling** (e.g. “Benign”, “Malignant”) and document **class-index ↔ label** mapping explicitly.  
- Automate **merging DCGAN outputs** into the classification dataset with a small script.  
- Add **tests**, **Dockerfile**, and a proper **LICENSE** file.

---

## 🤝 Contributing

1. Fork the repository and create a feature branch.  
2. Keep changes focused; match existing style in notebooks and `app.py`.  
3. Document any new paths, dataset layout, or model file names in this README.  
4. Open a pull request with a clear description of what was tested.

---

## 📄 License

No `LICENSE` file is present in this repository. **Suggested:** add an **MIT License** (or another license chosen by the authors) before public distribution.

---

## 👥 Authors

**Team Alphabet.io** — AVCOE, Sangamner  

- Suraj Zaware  
- Sagar Waghmare  
- Vishal Wandekar  
- Dipashri Shinde  
- Vedashri Waman  
