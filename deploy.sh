#!/bin/bash

# Exit script on any error
set -e

# Variables
PROJECT_NAME="<project_name>"
REGION="us-east-1" # Change this to your desired AWS region
S3_BUCKET="<bucket_name>"
EC2_KEY_PAIR_NAME="${PROJECT_NAME}-keypair"
SECURITY_GROUP_NAME="${PROJECT_NAME}-sg"
INSTANCE_TYPE="t2.micro" # Adjust based on your needs
AMI_ID="ami-0c02fb55956c7d316" # Amazon Linux 2 AMI ID (change for your region)
GITHUB_REPO_URL="<github_repo_url>"
APP_DIR="~/flask_app"
ENV_YML_FILE="current_environment.yml"
OLLAMA_VERSION="latest"
LLAMA_MODEL="llama-3.1"

# Install dependencies
echo "Installing dependencies..."
sudo apt-get update && sudo apt-get install -y awscli unzip python3 python3-pip git wget curl

# Create an S3 bucket for the project
echo "Creating S3 bucket..."
aws s3 mb s3://$S3_BUCKET --region $REGION

# Create a key pair for EC2 instances
echo "Creating EC2 key pair..."
aws ec2 create-key-pair --key-name $EC2_KEY_PAIR_NAME --query 'KeyMaterial' --output text > "${EC2_KEY_PAIR_NAME}.pem"
chmod 400 "${EC2_KEY_PAIR_NAME}.pem"

# Create a security group
echo "Creating security group..."
SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name $SECURITY_GROUP_NAME --description "Security group for Flask app" --query 'GroupId' --output text)

# Allow SSH and HTTP access
echo "Configuring security group rules..."
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 5000 --cidr 0.0.0.0/0

# Launch an EC2 instance
echo "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances --image-id $AMI_ID --count 1 --instance-type $INSTANCE_TYPE --key-name $EC2_KEY_PAIR_NAME --security-group-ids $SECURITY_GROUP_ID --query 'Instances[0].InstanceId' --output text)

echo "Waiting for instance to be in running state..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "Instance is running. Public IP: $PUBLIC_IP"

# Set up Flask app, Conda, and Ollama on the EC2 instance
echo "Setting up Flask app, Conda, and Ollama on EC2 instance..."
ssh -o "StrictHostKeyChecking no" -i "${EC2_KEY_PAIR_NAME}.pem" ec2-user@$PUBLIC_IP << EOF
    # Update system packages
    sudo yum update -y

    # Install necessary packages
    sudo yum install -y python3 pip git wget curl

    # Install Miniconda
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p \$HOME/miniconda
    export PATH="\$HOME/miniconda/bin:\$PATH"
    conda init

    # Create Conda environment from YAML file
    echo "Setting up Conda environment..."
    conda env create -f $ENV_YML_FILE

    # Activate the new Conda environment
    echo "Activating Conda environment..."
    source activate $(head -n 1 $ENV_YML_FILE | cut -d ' ' -f 2)

    # Clone the GitHub repository
    echo "Cloning GitHub repository..."
    git clone $GITHUB_REPO_URL $APP_DIR
    cd $APP_DIR

    # Install Flask app dependencies
    pip install -r requirements.txt

    # Install Ollama
    echo "Installing Ollama..."
    curl -s https://ollama.com/download/linux | bash
    sudo mv ollama /usr/local/bin

    # Download the Llama 3.1 model
    echo "Downloading Llama 3.1 model..."
    ollama pull $LLAMA_MODEL

    # Start the Flask app
    echo "Starting Flask app..."
    python3 app.py &
EOF

echo "Flask app is now running with Llama 3.1 and Conda environment set up."
echo "Access it at http://$PUBLIC_IP:5000"
