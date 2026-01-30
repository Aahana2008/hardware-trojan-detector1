# Hardware Trojan Detection System

AI-Powered RTL Security Analysis for detecting hardware trojans in Verilog code.

## 🚀 Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run trojan_detector_professional.py
```

## 🌐 Deploy to Streamlit Cloud (FREE)

### Step 1: Prepare GitHub Repository

1. Create a new repository on GitHub
2. Upload these files:
   - `trojan_detector_professional.py`
   - `requirements.txt`
   - `README.md`

### Step 2: Deploy

1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Choose `trojan_detector_professional.py` as the main file
6. Click "Deploy"

### Step 3: Get Your URL

Your app will be available at:
- `https://[your-app-name].streamlit.app`

Example: `https://hardware-trojan-detector.streamlit.app`

## 🎯 Custom Domain Setup

### Option 1: Using Streamlit Cloud
1. Buy a domain (e.g., from Namecheap, GoDaddy)
2. In Streamlit Cloud settings, add custom domain
3. Update DNS records:
   - Type: CNAME
   - Name: www (or @)
   - Value: [your-app].streamlit.app

### Option 2: Self-Hosting
Deploy on your own server (DigitalOcean, AWS, etc.)

## 📋 Features

- 🔍 Multi-file Verilog code analysis
- 🤖 8 ML models for trojan detection
- 📊 Interactive visualizations
- 🎯 62 feature extraction
- 💯 92% accuracy with KNN model

## 🛠️ Technical Details

### Models Supported:
- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- AdaBoost
- Gradient Boosting
- Random Forest
- Decision Tree
- Logistic Regression
- Support Vector Machine (SVM)

### Feature Categories:
- Structural features (11)
- Control flow features (5)
- Operator features (22)
- Clock/timing features (3)
- Suspicious patterns (7)
- Additional metrics (14)

## 📊 Performance

- **Accuracy:** 92.0%
- **F1 Score:** 95.8%
- **Recall:** 100%
- **ROC-AUC:** 95.7%

## 📝 Usage

1. **Upload Verilog Files:** Select multiple .v or .sv files
2. **Paste Code:** Directly paste Verilog code
3. **Analyze:** Get instant trojan detection results
4. **Train Models:** Compare different ML algorithms
5. **View Performance:** Analyze model metrics

## 🔒 Security Note

This tool is for educational and research purposes. Always verify results with manual code review for production systems.

## 👥 Authors

Hardware Security Research Team

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a pull request.

## 📧 Support

For questions or support, please open an issue on GitHub.
