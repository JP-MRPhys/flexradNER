
---

# **Flask App Deployment on AWS with GPU Support**

This repository contains a Flask application configured for deployment on an AWS GPU-enabled instance. The deployment includes setting up necessary dependencies, Conda environment management, and support for the Llama 3.1 model using Ollama.

---

## **Features**
- Deployment to a **`g4dn.xlarge`** AWS GPU instance.
- Integration with **Ollama** for AI model management.
- Automatic installation of NVIDIA drivers for GPU acceleration.
- Easy setup of a Conda environment using a `.yml` file.
- Flask app accessible via the public IP of the EC2 instance.

---

## **Deployment Prerequisites**
1. **AWS Account**:
   - An active AWS account with access to create EC2 instances, S3 buckets, and security groups.
2. **Dependencies**:
   - Ensure the following are installed locally:
     - `awscli` (configured with your AWS credentials).
     - `bash` for running the script.
3. **Conda Environment File**:
   - Ensure `current_environment.yml` is in the repository root.
   - Contains necessary dependencies for the Flask app and Llama model.
4. **GitHub Repository**:
   - Clone or fork this repository.

---

## **Steps to Deploy**

### 1. **Clone the Repository**
```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. **Update Deployment Variables**
In the `deploy.sh` script, update the following:
- `PROJECT_NAME`: Unique name for your project.
- `REGION`: Desired AWS region (default: `us-east-1`).
- `S3_BUCKET`: A unique name for the S3 bucket.
- `GITHUB_REPO_URL`: URL of this repository.
- `ENV_YML_FILE`: Name of the Conda environment file (default: `current_environment.yml`).

### 3. **Run the Deployment Script**
Make the script executable:
```bash
chmod +x deploy.sh
```

Run the script:
```bash
./deploy.sh
```

### 4. **Access Your Application**
After deployment, the script will display the public IP of your EC2 instance. Open a browser and navigate to:
```
http://<public_ip>:5000
```

---

## **File Structure**
```plaintext
.
├── app.py                # Flask application entry point
├── requirements.txt      # Python dependencies for Flask app
├── deploy.sh             # Deployment script for AWS
├── current_environment.yml # Conda environment configuration file
└── README.md             # Deployment documentation (this file)
```

---

## **Technical Details**

### **AWS GPU Instance**
- **Instance Type**: `g4dn.xlarge` (NVIDIA T4 Tensor Core GPU).
- **AMI**: Amazon Linux 2.

### **NVIDIA Driver Setup**
- Installs the necessary NVIDIA drivers and CUDA dependencies for GPU acceleration.

### **Conda Environment**
- Automatically created and activated using `current_environment.yml`.

### **Ollama and Llama 3.1**
- Installs Ollama for AI model management.
- Downloads the Llama 3.1 model for use in the Flask app.

---

## **Troubleshooting**
### Common Issues:
1. **Permission Denied for Key Pair**:
   - Ensure the `.pem` file is secured:
     ```bash
     chmod 400 <keypair_name>.pem
     ```

2. **Public IP Not Accessible**:
   - Check your security group rules to allow access on port `5000` and `22` (SSH).

3. **Conda Environment Errors**:
   - Verify that the `current_environment.yml` file is correctly formatted.

---

## **Future Improvements**
- Add HTTPS support with an SSL certificate.
- Enable auto-scaling for the application.
- Configure CI/CD for automated deployments.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

If you have any issues or questions, please feel free to open an issue in the repository. 😊

--- 

Let me know if you'd like further tweaks!