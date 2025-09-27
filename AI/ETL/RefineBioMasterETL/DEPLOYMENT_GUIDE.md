# Production ETL Pipeline Deployment Guide

This guide provides step-by-step instructions for deploying the production-grade bioinformatics ETL pipeline in various environments.

## 🎯 Deployment Options

### 1. Local Development Deployment
### 2. Server/VM Deployment
### 3. Docker Container Deployment
### 4. Cloud Platform Deployment (AWS, Azure, GCP)

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended), Windows Server 2019+, or macOS
- **Memory**: 8GB RAM minimum, 16GB+ recommended for large datasets
- **Storage**: 50GB+ available space for data and logs
- **Network**: Stable internet connection for database access

### Software Dependencies
- Python 3.8+ with pip
- SQL Server 2017+ or compatible database
- Git for version control
- Docker (optional, for containerized deployment)

## 🔧 Local Development Deployment

### Step 1: Environment Setup

```bash
# Create project directory
mkdir -p /opt/etl-pipeline
cd /opt/etl-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Clone repository
git clone <repository-url> .
```

### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install additional development tools
pip install pytest pytest-cov black flake8 mypy
```

### Step 3: Database Setup

```bash
# Install SQL Server (Ubuntu example)
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
sudo add-apt-repository "$(wget -qO- https://packages.microsoft.com/config/ubuntu/20.04/mssql-server-2022.list)"
sudo apt-get update
sudo apt-get install -y mssql-server

# Configure SQL Server
sudo /opt/mssql/bin/mssql-conf setup

# Install SQL command line tools
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list | sudo tee /etc/apt/sources.list.d/msprod.list
sudo apt-get update
sudo apt-get install mssql-tools unixodbc-dev

# Add SQL tools to PATH
echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Database Schema Creation

```bash
# Create database
sqlcmd -S localhost -U sa -P YourPassword -Q "CREATE DATABASE BioinformaticsWarehouse"

# Run schema setup
sqlcmd -S localhost -U sa -P YourPassword -d BioinformaticsWarehouse -i robust_database_setup.sql
```

### Step 5: Configuration

```bash
# Create environment file
cat > .env << EOF
ETL_BASE_PATH=/opt/etl-pipeline/data
DB_CONNECTION_STRING=Server=localhost;Database=BioinformaticsWarehouse;User Id=sa;Password=YourPassword;
ETL_LOG_LEVEL=INFO
ETL_LOG_FILE=/opt/etl-pipeline/logs/etl.log
ETL_BATCH_SIZE=25000
ETL_MAX_WORKERS=4
ETL_MEMORY_LIMIT_MB=4096
EOF

# Create directories
mkdir -p /opt/etl-pipeline/data
mkdir -p /opt/etl-pipeline/logs
mkdir -p /opt/etl-pipeline/reports

# Set permissions
chmod 755 /opt/etl-pipeline
chmod 644 .env
```

### Step 6: Test Deployment

```bash
# Run test suite
python test_etl_pipeline.py

# Test with sample data
python robust_main_etl.py --validate-only
```

## 🖥️ Server/VM Deployment

### Step 1: Server Preparation

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y python3-pip python3-venv git curl wget

# Create etl user
sudo useradd -m -s /bin/bash etl
sudo usermod -aG sudo etl
```

### Step 2: System Optimization

```bash
# Increase file descriptor limits
echo "etl soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "etl hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Optimize memory settings
echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p

# Configure firewall (if needed)
sudo ufw allow 1433/tcp  # SQL Server
sudo ufw allow 22/tcp    # SSH
```

### Step 3: Install and Configure Database

```bash
# Switch to etl user
sudo su - etl

# Follow database setup steps from Local Development section
# ...
```

### Step 4: Create System Service

```bash
# Create systemd service file
sudo tee /etc/systemd/system/etl-pipeline.service << EOF
[Unit]
Description=Bioinformatics ETL Pipeline
After=network.target mssql-server.service
Wants=mssql-server.service

[Service]
Type=oneshot
User=etl
Group=etl
WorkingDirectory=/opt/etl-pipeline
Environment=PATH=/opt/etl-pipeline/venv/bin
EnvironmentFile=/opt/etl-pipeline/.env
ExecStart=/opt/etl-pipeline/venv/bin/python robust_main_etl.py
StandardOutput=journal
StandardError=journal
SyslogIdentifier=etl-pipeline

[Install]
WantedBy=multi-user.target
EOF

# Create timer for scheduled execution
sudo tee /etc/systemd/system/etl-pipeline.timer << EOF
[Unit]
Description=Run ETL Pipeline Daily
Requires=etl-pipeline.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable services
sudo systemctl daemon-reload
sudo systemctl enable etl-pipeline.timer
sudo systemctl start etl-pipeline.timer
```

### Step 5: Monitoring Setup

```bash
# Install monitoring tools
sudo apt-get install -y htop iotop nethogs

# Create monitoring script
cat > /opt/etl-pipeline/monitor_pipeline.py << 'EOF'
#!/usr/bin/env python3
import psutil
import logging
from datetime import datetime

def monitor_system():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    print(f"{datetime.now()}: CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")

if __name__ == "__main__":
    monitor_system()
EOF

chmod +x /opt/etl-pipeline/monitor_pipeline.py

# Add to crontab
echo "*/5 * * * * /opt/etl-pipeline/venv/bin/python /opt/etl-pipeline/monitor_pipeline.py >> /opt/etl-pipeline/logs/system_monitor.log" | crontab -
```

## 🐳 Docker Deployment

### Step 1: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install SQL Server ODBC driver
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -s /bin/bash etl
RUN chown -R etl:etl /app
USER etl

# Set environment variables
ENV PYTHONPATH=/app
ENV ETL_LOG_LEVEL=INFO

# Create data directory
RUN mkdir -p /app/data /app/logs /app/reports

# Default command
CMD ["python", "robust_main_etl.py"]
```

### Step 2: Create Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  sqlserver:
    image: mcr.microsoft.com/mssql/server:2022-latest
    environment:
      ACCEPT_EULA: Y
      SA_PASSWORD: YourStrongPassword123!
      MSSQL_PID: Developer
    ports:
      - "1433:1433"
    volumes:
      - sqlserver_data:/var/opt/mssql
    networks:
      - etl-network

  etl-pipeline:
    build: .
    depends_on:
      - sqlserver
    environment:
      ETL_BASE_PATH: /app/data
      DB_CONNECTION_STRING: Server=sqlserver;Database=BioinformaticsWarehouse;User Id=sa;Password=YourStrongPassword123!;
      ETL_LOG_LEVEL: INFO
      ETL_BATCH_SIZE: 25000
      ETL_MAX_WORKERS: 4
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./reports:/app/reports
    networks:
      - etl-network
    restart: unless-stopped

  monitoring:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - etl-network

volumes:
  sqlserver_data:
  grafana_data:

networks:
  etl-network:
    driver: bridge
```

### Step 3: Build and Deploy

```bash
# Build Docker image
docker build -t etl-pipeline:latest .

# Deploy with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f etl-pipeline
```

## ☁️ Cloud Platform Deployment

### AWS Deployment

#### Step 1: EC2 Instance Setup

```bash
# Launch EC2 instance (Ubuntu 20.04, t3.large or larger)
# Configure security groups:
# - Inbound: SSH (22), SQL Server (1433), HTTP (80)
# - Outbound: All traffic

# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Follow server deployment steps
```

#### Step 2: RDS Database Setup

```bash
# Create RDS SQL Server instance
aws rds create-db-instance \
    --db-instance-identifier etl-database \
    --db-instance-class db.t3.medium \
    --engine sqlserver-ex \
    --master-username admin \
    --master-user-password YourStrongPassword123! \
    --allocated-storage 100 \
    --vpc-security-group-ids your-sg-id

# Update connection string in .env
DB_CONNECTION_STRING=Server=your-rds-endpoint;Database=BioinformaticsWarehouse;User Id=admin;Password=YourStrongPassword123!;
```

#### Step 3: S3 Data Storage

```bash
# Create S3 bucket for data
aws s3 mb s3://your-etl-data-bucket

# Upload sample data
aws s3 cp your-data/ s3://your-etl-data-bucket/ --recursive

# Configure IAM role for EC2 to access S3
aws iam create-role --role-name etl-s3-access --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name etl-s3-access --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
```

### Azure Deployment

#### Step 1: Azure SQL Database

```bash
# Create Azure SQL Database
az sql server create \
    --name etl-sql-server \
    --resource-group your-resource-group \
    --location eastus \
    --admin-user admin \
    --admin-password YourStrongPassword123!

# Configure firewall
az sql server firewall-rule create \
    --name AllowYourIP \
    --server etl-sql-server \
    --resource-group your-resource-group \
    --start-ip-address your-ip \
    --end-ip-address your-ip
```

#### Step 2: Azure Virtual Machine

```bash
# Create VM
az vm create \
    --resource-group your-resource-group \
    --name etl-vm \
    --image UbuntuLTS \
    --admin-username azureuser \
    --size Standard_B2s \
    --generate-ssh-keys

# Install Azure CLI on VM
az vm run-command invoke \
    --resource-group your-resource-group \
    --name etl-vm \
    --command-id RunShellScript \
    --scripts "curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
```

### GCP Deployment

#### Step 1: Cloud SQL Setup

```bash
# Create Cloud SQL instance
gcloud sql instances create etl-instance \
    --database-version=SQLSERVER_2019_STANDARD \
    --tier=db-custom-2-3840 \
    --region=us-central1 \
    --root-password=YourStrongPassword123!

# Create database
gcloud sql databases create BioinformaticsWarehouse --instance=etl-instance
```

#### Step 2: Compute Engine

```bash
# Create VM instance
gcloud compute instances create etl-vm \
    --zone=us-central1-a \
    --machine-type=e2-medium \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB
```

## 📊 Monitoring and Alerting

### Setting Up Monitoring

#### 1. Database Monitoring

```sql
-- Create monitoring views
CREATE VIEW monitoring.etl_performance_summary AS
SELECT 
    study_code,
    COUNT(*) as execution_count,
    AVG(duration_seconds) as avg_duration,
    AVG(records_processed) as avg_records,
    AVG(validation_score) as avg_quality,
    SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as success_count,
    SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) as failure_count
FROM audit.etl_execution_log
WHERE start_time >= DATEADD(day, -7, GETDATE())
GROUP BY study_code;
```

#### 2. System Monitoring Script

```python
# monitoring/health_check.py
import psutil
import requests
import logging
from datetime import datetime

class ETLHealthMonitor:
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url
        self.logger = logging.getLogger(__name__)
    
    def check_system_resources(self):
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        alerts = []
        
        if cpu_percent > 80:
            alerts.append(f"High CPU usage: {cpu_percent}%")
        
        if memory.percent > 80:
            alerts.append(f"High memory usage: {memory.percent}%")
        
        if disk.percent > 90:
            alerts.append(f"Low disk space: {disk.percent}%")
        
        return alerts
    
    def check_database_connection(self):
        try:
            # Test database connection
            import pyodbc
            conn = pyodbc.connect(os.getenv('DB_CONNECTION_STRING'))
            conn.close()
            return True
        except Exception as e:
            return False
    
    def send_alert(self, message):
        if self.webhook_url:
            try:
                requests.post(self.webhook_url, json={
                    "text": f"ETL Pipeline Alert: {message}"
                })
            except Exception as e:
                self.logger.error(f"Failed to send alert: {e}")

# Usage in cron
monitor = ETLHealthMonitor(webhook_url="your-slack-webhook")
alerts = monitor.check_system_resources()
for alert in alerts:
    monitor.send_alert(alert)
```

#### 3. Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "ETL Pipeline Monitoring",
    "panels": [
      {
        "title": "ETL Execution Status",
        "type": "stat",
        "targets": [
          {
            "expr": "SELECT COUNT(*) as executions, AVG(duration_seconds) as avg_duration FROM audit.etl_execution_log WHERE start_time >= DATEADD(hour, -24, GETDATE())"
          }
        ]
      },
      {
        "title": "Data Quality Score",
        "type": "timeseries",
        "targets": [
          {
            "expr": "SELECT study_code, validation_score, start_time FROM audit.etl_execution_log WHERE start_time >= DATEADD(day, -7, GETDATE()) ORDER BY start_time"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Best Practices

### 1. Database Security

```sql
-- Create dedicated ETL user
CREATE LOGIN etl_user WITH PASSWORD = 'YourStrongPassword123!';
CREATE USER etl_user FOR LOGIN etl_user;

-- Grant minimum required permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA::dbo TO etl_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA::staging TO etl_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON SCHEMA::audit TO etl_user;
GRANT EXECUTE ON SCHEMA::dbo TO etl_user;

-- Create read-only user for monitoring
CREATE LOGIN monitoring_user WITH PASSWORD = 'AnotherStrongPassword123!';
CREATE USER monitoring_user FOR LOGIN monitoring_user;
GRANT SELECT ON SCHEMA::dbo TO monitoring_user;
GRANT SELECT ON SCHEMA::audit TO monitoring_user;
```

### 2. Application Security

```python
# config/security.py
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.encryption_key = os.getenv('ETL_ENCRYPTION_KEY', Fernet.generate_key())
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_connection_string(self, conn_str):
        return self.cipher.encrypt(conn_str.encode())
    
    def decrypt_connection_string(self, encrypted_conn_str):
        return self.cipher.decrypt(encrypted_conn_str).decode()

# Usage
secure_config = SecureConfig()
encrypted_conn_str = secure_config.encrypt_connection_string("your_connection_string")
```

### 3. Network Security

```bash
# Configure iptables for database protection
sudo iptables -A INPUT -p tcp --dport 1433 -s 10.0.0.0/8 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 1433 -s 172.16.0.0/12 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 1433 -s 192.168.0.0/16 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 1433 -j DROP
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. Memory Issues
```bash
# Check memory usage
free -h
top -p $(pgrep -f "python.*etl")

# Reduce batch size
export ETL_BATCH_SIZE=10000
export ETL_CHUNK_SIZE=5000
```

#### 2. Database Connection Issues
```bash
# Test connection
sqlcmd -S your_server -U your_user -P your_password -Q "SELECT 1"

# Check network connectivity
telnet your_server 1433

# Verify firewall rules
sudo ufw status
```

#### 3. Performance Issues
```bash
# Monitor system resources
iostat -x 1
vmstat 1

# Check database performance
sqlcmd -S your_server -Q "SELECT * FROM sys.dm_exec_requests WHERE status = 'running'"
```

#### 4. Data Validation Failures
```bash
# Run validation-only mode
python robust_main_etl.py --validate-only --studies SRP049820

# Check validation report
cat reports/validation_report_*.txt
```

### Log Analysis

```bash
# Search for errors
grep -i "error" logs/etl.log | tail -50

# Performance analysis
grep "duration" logs/etl.log | awk '{print $NF}' | sort -n

# Memory usage analysis
grep "memory" logs/etl.log | awk '{print $6}' | sort -n
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks

1. **Daily**
   - Monitor pipeline execution logs
   - Check system resource usage
   - Verify data quality scores

2. **Weekly**
   - Review performance metrics
   - Clean up old log files
   - Update dependencies

3. **Monthly**
   - Database maintenance (index rebuild, statistics update)
   - Security patch updates
   - Capacity planning review

### Backup Strategy

```bash
# Database backup
sqlcmd -S your_server -Q "BACKUP DATABASE BioinformaticsWarehouse TO DISK = '/backup/etl_db_$(date +%Y%m%d).bak'"

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env *.py *.sql

# Log backup
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

### Disaster Recovery

1. **Database Recovery**
   ```bash
   sqlcmd -S your_server -Q "RESTORE DATABASE BioinformaticsWarehouse FROM DISK = '/backup/etl_db_latest.bak'"
   ```

2. **Application Recovery**
   ```bash
   # Restore from backup
   git clone <backup-repository> /opt/etl-pipeline-recovery
   cd /opt/etl-pipeline-recovery
   pip install -r requirements.txt
   ```

## 📈 Scaling Considerations

### Horizontal Scaling
- **Multiple Instances**: Run parallel ETL instances for different studies
- **Load Balancing**: Distribute processing across multiple servers
- **Queue-Based Processing**: Use message queues for task distribution

### Vertical Scaling
- **Memory**: Increase RAM for larger batch processing
- **CPU**: More cores for parallel processing
- **Storage**: Faster SSDs for I/O-intensive operations

### Database Scaling
- **Partitioning**: Partition large tables by study or date
- **Sharding**: Distribute data across multiple databases
- **Read Replicas**: Separate read and write operations

---

This deployment guide provides comprehensive instructions for various deployment scenarios. For specific questions or issues, please refer to the troubleshooting section or create an issue in the repository.