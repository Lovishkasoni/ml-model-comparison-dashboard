const API_BASE = 'http://localhost:5000';

let uploadedColumns = [];
let trainedModels = [];

// DOM Elements
const csvFile = document.getElementById('csvFile');
const fileLabel = document.querySelector('.file-label');
const targetColumn = document.getElementById('targetColumn');
const uploadBtn = document.getElementById('uploadBtn');
const uploadStatus = document.getElementById('uploadStatus');
const trainBtn = document.getElementById('trainBtn');
const trainingStatus = document.getElementById('trainingStatus');
const resultsTable = document.getElementById('resultsTable');
const bestModelDiv = document.getElementById('bestModel');
const featureModelSelect = document.getElementById('featureModelSelect');
const tuneModelSelect = document.getElementById('tuneModelSelect');
const featureImportancePlot = document.getElementById('featureImportancePlot');
const tuningParamsDiv = document.getElementById('tuningParams');
const tuneBtn = document.getElementById('tuneBtn');
const tuningResults = document.getElementById('tuningResults');

// File Upload Handler
csvFile.addEventListener('change', function(e) {
    const fileName = e.target.files[0]?.name || 'Select file...';
    fileLabel.innerHTML = `<span>📄 ${fileName}</span>`;
});

// Upload Button
uploadBtn.addEventListener('click', async function() {
    const file = csvFile.files[0];
    const target = targetColumn.value;

    if (!file || !target) {
        showStatus(uploadStatus, 'Please select both file and target column', 'error');
        return;
    }

    uploadBtn.disabled = true;
    showStatus(uploadStatus, 'Uploading and preprocessing...', 'info');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_column', target);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error);
        }

        showStatus(uploadStatus, `✅ Data preprocessed! Problem type: ${data.problem_type}`, 'success');
        
        // Show training section
        document.querySelector('.training-section').style.display = 'block';
        
        // Populate model selectors
        trainedModels = [];

    } catch (error) {
        showStatus(uploadStatus, `Error: ${error.message}`, 'error');
    } finally {
        uploadBtn.disabled = false;
    }
});

// Train Button
trainBtn.addEventListener('click', async function() {
    trainBtn.disabled = true;
    showStatus(trainingStatus, 'Training all models... This may take a moment...', 'info');

    try {
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST'
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error);
        }

        showStatus(trainingStatus, '✅ Training completed!', 'success');
        
        // Display results
        displayResults(data.results, data.best_model);
        trainedModels = Object.keys(data.results);

        // Populate model selectors
        populateModelSelectors();

        // Show results section
        document.querySelector('.results-section').style.display = 'block';

    } catch (error) {
        showStatus(trainingStatus, `Error: ${error.message}`, 'error');
    } finally {
        trainBtn.disabled = false;
    }
});

// Display Results
function displayResults(results, bestModel) {
    // Build table
    let tableHTML = '<table><thead><tr><th>Model</th>';
    
    if (results[Object.keys(results)[0]]) {
        const metrics = Object.keys(results[Object.keys(results)[0]]);
        metrics.forEach(metric => {
            tableHTML += `<th>${metric}</th>`;
        });
    }
    
    tableHTML += '</tr></thead><tbody>';

    for (const [model, metrics] of Object.entries(results)) {
        tableHTML += `<tr><td><strong>${model}</strong></td>`;
        for (const [metric, value] of Object.entries(metrics)) {
            const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
            tableHTML += `<td>${formattedValue}</td>`;
        }
        tableHTML += '</tr>';
    }

    tableHTML += '</tbody></table>';
    resultsTable.innerHTML = tableHTML;

    // Display best model
    bestModelDiv.innerHTML = `<strong>${bestModel}</strong>`;
}

// Populate Model Selectors
function populateModelSelectors() {
    const modelOptions = trainedModels.map(model => 
        `<option value="${model}">${model}</option>`
    ).join('');

    featureModelSelect.innerHTML = modelOptions;
    tuneModelSelect.innerHTML = modelOptions;

    // Load feature importance for first model
    if (trainedModels.length > 0) {
        loadFeatureImportance(trainedModels[0]);
        loadTuningParams(trainedModels[0]);
    }
}

// Load Feature Importance
async function loadFeatureImportance(modelName) {
    try {
        const response = await fetch(`${API_BASE}/feature-importance/${modelName}`);
        const data = await response.json();

        if (response.ok) {
            plotFeatureImportance(data.features, data.importance);
        }
    } catch (error) {
        console.error('Error loading feature importance:', error);
    }
}

// Plot Feature Importance
function plotFeatureImportance(features, importance) {
    const trace = {
        y: features,
        x: importance,
        type: 'bar',
        orientation: 'h',
        marker: {
            color: 'rgba(102, 126, 234, 0.8)'
        }
    };

    const layout = {
        title: 'Feature Importance',
        xaxis: { title: 'Importance' },
        yaxis: { title: 'Features' },
        margin: { l: 150 }
    };

    Plotly.newPlot(featureImportancePlot, [trace], layout);
}

// Load Tuning Parameters
function loadTuningParams(modelName) {
    const params = getHyperparametersForModel(modelName);
    
    let paramsHTML = '';
    for (const [paramName, paramValue] of Object.entries(params)) {
        paramsHTML += `
            <div class="param-group">
                <label for="param_${paramName}">${paramName}:</label>
                <input 
                    type="text" 
                    id="param_${paramName}" 
                    value="${paramValue}" 
                    placeholder="Enter value"
                />
            </div>
        `;
    }

    tuningParamsDiv.innerHTML = paramsHTML;
}

// Get Hyperparameters for Model
function getHyperparametersForModel(modelName) {
    const defaults = {
        'Random Forest': { 'n_estimators': 100, 'max_depth': 10 },
        'SVM': { 'C': 1.0, 'kernel': 'rbf' },
        'XGBoost': { 'n_estimators': 100, 'learning_rate': 0.1 },
        'KNN': { 'n_neighbors': 5 },
        'Decision Tree': { 'max_depth': 10 },
        'Logistic Regression': { 'C': 1.0 },
        'Linear Regression': {}
    };

    return defaults[modelName] || {};
}

// Feature Model Selector Change
featureModelSelect.addEventListener('change', function() {
    loadFeatureImportance(this.value);
});

// Tune Model Selector Change
tuneModelSelect.addEventListener('change', function() {
    loadTuningParams(this.value);
});

// Tune Button
tuneBtn.addEventListener('click', async function() {
    const modelName = tuneModelSelect.value;
    const params = {};

    // Collect parameters
    document.querySelectorAll('#tuningParams input').forEach(input => {
        const paramName = input.id.replace('param_', '');
        let value = input.value;

        // Try to convert to appropriate type
        if (!isNaN(value) && value !== '') {
            value = value.includes('.') ? parseFloat(value) : parseInt(value);
        }

        params[paramName] = value;
    });

    tuneBtn.disabled = true;
    showStatus(tuningResults, 'Retraining with new parameters...', 'info');

    try {
        const response = await fetch(`${API_BASE}/tune`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: modelName,
                params: params
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error);
        }

        let resultsText = `✅ Model retrained successfully!<br><strong>${modelName}</strong><br>`;
        for (const [metric, value] of Object.entries(data.metrics)) {
            const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
            resultsText += `${metric}: ${formattedValue}<br>`;
        }

        showStatus(tuningResults, resultsText, 'success');

    } catch (error) {
        showStatus(tuningResults, `Error: ${error.message}`, 'error');
    } finally {
        tuneBtn.disabled = false;
    }
});

// Utility: Show Status Message
function showStatus(element, message, type) {
    element.innerHTML = message;
    element.className = `status-message ${type}`;
}

// Initialize Target Column Options on File Select
csvFile.addEventListener('change', async function(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(event) {
        const csv = event.target.result;
        const rows = csv.split('\n');
        const headers = rows[0].split(',').map(h => h.trim());

        uploadedColumns = headers;
        targetColumn.innerHTML = '<option value="">Select target column...</option>';

        headers.forEach(header => {
            if (header) {
                const option = document.createElement('option');
                option.value = header;
                option.textContent = header;
                targetColumn.appendChild(option);
            }
        });

        targetColumn.disabled = false;
        uploadBtn.disabled = false;
    };

    reader.readAsText(file);
});