pipeline {
    agent any
    
    environment {
        // GitHub Configuration
        GITHUB_REPO = 'ThuanNaN/aio2025-mlops-cicd'
        GITHUB_TOKEN = credentials('github-token')
        
        // Conda environment name (pre-created)
        CONDA_ENV = 'mlops-cicd'
        
        // DVC Configuration
        DVC_REMOTE = 'local'
    }
    
    triggers {
        // Generic Webhook Trigger - triggers on DVC data changes
        genericTrigger(
            genericVariables: [
                [key: 'ref', value: '$.ref'],
                [key: 'commits', value: '$.commits'],
                [key: 'repository', value: '$.repository.full_name'],
                [key: 'data_changed', value: '$.data_changed', defaultValue: 'false']
            ],
            causeString: 'Triggered by DVC data change webhook',
            token: 'dvc-data-change-trigger',
            printContributedVariables: true,
            printPostContent: true,
            silentResponse: false,
            regexpFilterText: '$data_changed',
            regexpFilterExpression: 'true'
        )
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
                sh '''
                    echo "=== Repository Info ==="
                    git log -1 --pretty=format:"%H - %s (%an, %ad)"
                    echo ""
                '''
            }
        }
        
        stage('Setup Python Environment') {
            steps {
                sh '''
                    echo "=== Using pre-created Conda Environment ==="
                    eval "$(conda shell.bash hook)"
                    conda activate ${CONDA_ENV}
                    python --version
                    echo "Environment ready!"
                '''
            }
        }
        
        stage('DVC Pull Data') {
            steps {
                sh '''
                    echo "=== Pulling DVC Data ==="
                    eval "$(conda shell.bash hook)"
                    conda activate ${CONDA_ENV}
                    dvc pull -f || echo "No remote data to pull, using local data"
                '''
            }
        }
        
        stage('Check Data Changes') {
            steps {
                script {
                    def dataChanged = sh(
                        script: '''
                            eval "$(conda shell.bash hook)"
                            conda activate ${CONDA_ENV}
                            dvc status 2>/dev/null | grep -q "changed" && echo "true" || echo "false"
                        ''',
                        returnStdout: true
                    ).trim()
                    
                    env.DATA_CHANGED = dataChanged
                    echo "Data changed: ${dataChanged}"
                }
            }
        }
        
        stage('Train Model') {
            when {
                anyOf {
                    expression { env.DATA_CHANGED == 'true' }
                    expression { params.FORCE_TRAIN == true }
                    triggeredBy 'GenericTrigger'
                }
            }
            steps {
                sh '''
                    echo "=== Training YOLO Model ==="
                    eval "$(conda shell.bash hook)"
                    conda activate ${CONDA_ENV}
                    
                    # Run DVC pipeline (includes training)
                    dvc repro train --force
                    
                    echo "=== Training Complete ==="
                '''
            }
            post {
                success {
                    archiveArtifacts artifacts: 'runs/detect/train/weights/best.pt', fingerprint: true
                    archiveArtifacts artifacts: 'runs/detect/train/results.csv', fingerprint: true
                }
            }
        }
        
        stage('Push DVC Changes') {
            steps {
                sh '''
                    echo "=== Pushing DVC Changes ==="
                    eval "$(conda shell.bash hook)"
                    conda activate ${CONDA_ENV}
                    dvc push || echo "No changes to push"
                '''
            }
        }
        
        stage('Trigger GitHub Actions Deployment') {
            when {
                anyOf {
                    expression { env.DATA_CHANGED == 'true' }
                    expression { params.FORCE_TRAIN == true }
                }
            }
            steps {
                sh '''
                    echo "=== Triggering GitHub Actions Deployment ==="
                    
                    # Trigger repository_dispatch event to GitHub Actions
                    curl -X POST \
                        -H "Accept: application/vnd.github.v3+json" \
                        -H "Authorization: token ${GITHUB_TOKEN}" \
                        https://api.github.com/repos/${GITHUB_REPO}/dispatches \
                        -d '{"event_type": "model-updated", "client_payload": {"triggered_by": "jenkins", "model_version": "'$(date +%Y%m%d-%H%M%S)'"}}'
                    
                    echo "GitHub Actions deployment triggered successfully!"
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed!'
        }
    }
    
    parameters {
        booleanParam(name: 'FORCE_TRAIN', defaultValue: false, description: 'Force training even if no data changes detected')
    }
}
