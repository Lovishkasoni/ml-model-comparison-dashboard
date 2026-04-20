from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from data_processor import DataProcessor
from ml_models import MLModelComparison
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '../frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)

# Global variables to store state
data_processor = None
ml_comparison = None
current_data = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'OK'}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and preprocessing"""
    global data_processor, ml_comparison, current_data
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        target_column = request.form.get('target_column')
        
        if not target_column:
            return jsonify({'error': 'Target column not specified'}), 400
        
        # Load data
        data_processor = DataProcessor()
        df, error = data_processor.load_data(file)
        
        if error:
            return jsonify({'error': f'File load error: {error}'}), 400
        
        # Preprocess data
        result, error = data_processor.preprocess(df, target_column)
        
        if error:
            return jsonify({'error': f'Preprocessing error: {error}'}), 400
        
        current_data = data_processor.get_data()
        
        # Initialize ML comparison
        ml_comparison = MLModelComparison(problem_type=result['problem_type'])
        
        # Build message with removed features info
        message = 'Data uploaded and preprocessed successfully'
        if result.get('removed_features'):
            removed_str = ', '.join(result['removed_features'])
            message += f'\n\nRemoved non-predictive columns: {removed_str}'
        
        return jsonify({
            'success': True,
            'message': message,
            'problem_type': result['problem_type'],
            'n_features': result['n_features'],
            'n_samples': result['n_samples'],
            'features': result['features'],
            'removed_features': result.get('removed_features', [])
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train_models():
    """Train all models"""
    global ml_comparison, current_data
    
    try:
        if ml_comparison is None or current_data is None:
            return jsonify({'error': 'No data loaded. Please upload a file first.'}), 400
        
        results = ml_comparison.train_all_models(
            current_data['X_train'],
            current_data['X_test'],
            current_data['y_train'],
            current_data['y_test']
        )
        
        # Convert to JSON-serializable format
        results_json = {}
        for model_name, metrics in results.items():
            results_json[model_name] = {k: float(v) for k, v in metrics.items()}
        
        best_model = ml_comparison.get_best_model_name()
        
        return jsonify({
            'success': True,
            'results': results_json,
            'best_model': best_model
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Training error: {str(e)}'}), 500

@app.route('/feature-importance/<model_name>', methods=['GET'])
def get_feature_importance(model_name):
    """Get feature importance for a model"""
    global ml_comparison, current_data
    
    try:
        if ml_comparison is None:
            return jsonify({'error': 'No model trained'}), 400
        
        importance = ml_comparison.get_feature_importance(model_name)
        features = current_data['feature_names']
        
        # Sort by importance
        sorted_importance = sorted(
            zip(features, importance),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return jsonify({
            'success': True,
            'model': model_name,
            'features': [x[0] for x in sorted_importance],
            'importance': [float(x[1]) for x in sorted_importance]
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/tune', methods=['POST'])
def tune_hyperparameters():
    """Tune hyperparameters for a model"""
    global ml_comparison, current_data
    
    try:
        data = request.json
        model_name = data.get('model_name')
        params = data.get('params', {})
        
        if ml_comparison is None:
            return jsonify({'error': 'No model trained'}), 400
        
        metrics, error = ml_comparison.tune_hyperparameters(
            model_name,
            params,
            current_data['X_train'],
            current_data['X_test'],
            current_data['y_train'],
            current_data['y_test']
        )
        
        if error:
            return jsonify({'error': error}), 400
        
        metrics_json = {k: float(v) for k, v in metrics.items()}
        
        return jsonify({
            'success': True,
            'model': model_name,
            'metrics': metrics_json
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)