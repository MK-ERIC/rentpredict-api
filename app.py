"""
Kigali Apartment Price Prediction API
Final Year Capstone Project — Author: Mukesha
Best model auto-detected from Model/saved_models/ at startup.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
CORS(app)

# ── Load Saved Models ───────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'Model', 'saved_models')

print("=" * 50)
print(f"  Models folder: {MODELS_DIR}")
print(f"  Exists: {os.path.exists(MODELS_DIR)}")
if os.path.exists(MODELS_DIR):
    print(f"  Files: {os.listdir(MODELS_DIR)}")
print("=" * 50)

try:
    scaler         = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    feature_cols   = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.pkl'))
    best_file      = next(f for f in os.listdir(MODELS_DIR) if f.startswith('best_model_'))
    best_model     = joblib.load(os.path.join(MODELS_DIR, best_file))
    MODEL_NAME     = best_file.replace('best_model_', '').replace('.pkl', '').replace('_', ' ').title()
    print(f"[MODEL] Loaded: {best_file}  ({MODEL_NAME})")
except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    print(f"[ERROR] Make sure saved_models/ is at: {MODELS_DIR}")
    raise

# ── Constants ────────────────────────────────────────────────────
AMENITY_COLS = [
    'security', 'water', 'parking', 'wifi',
    'garden', 'swimming_pool', 'gym', 'air_conditioning'
]
REQUIRED_FIELDS = [
    'city', 'district', 'sector',
    'total_area', 'rooms', 'bathrooms',
    'year_built', 'floor_number',
    'distance_school', 'distance_hospital',
    'security', 'water', 'parking', 'wifi',
    'garden', 'swimming_pool', 'gym', 'air_conditioning'
]
VALID_CITIES    = ['Kigali']
VALID_DISTRICTS = ['Gasabo', 'Kicukiro', 'Nyarugenge']
VALID_SECTORS   = [
    'Gacuriro', 'Gatenga', 'Kacyiru', 'Kagarama', 'Kanombe',
    'Kimironko', 'Kiyovu', 'Muhima', 'Nyabugogo', 'Nyamirambo',
    'Nyarugunga', 'Nyarutarama', 'Remera',
]

def validate_input(data):
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return False, f"Missing fields: {missing}"
    if data['city'] not in VALID_CITIES:
        return False, f"city must be one of {VALID_CITIES}"
    if data['district'] not in VALID_DISTRICTS:
        return False, f"district must be one of {VALID_DISTRICTS}"
    if data['sector'] not in VALID_SECTORS:
        return False, f"sector must be one of {VALID_SECTORS}"
    for field in AMENITY_COLS:
        if int(float(data[field])) not in [0, 1]:
            return False, f"Field '{field}' must be 0 or 1"
    if not (1900 <= int(data['year_built']) <= 2026):
        return False, "year_built must be between 1900 and 2026"
    if float(data['total_area']) <= 0:
        return False, "total_area must be greater than 0"
    if int(data['rooms']) <= 0:
        return False, "rooms must be greater than 0"
    return True, None

def preprocess_input(data):
    df = pd.DataFrame([data])
    df['amenity_score'] = df[AMENITY_COLS].sum(axis=1)
    df['apartment_age'] = 2026 - df['year_built']
    df['area_per_room'] = df['total_area'] / df['rooms']
    df['total_rooms']   = df['rooms'] + df['bathrooms']
    df = df.drop(columns=['water', 'year_built'])
    for col in ['city', 'district', 'sector']:
        df[col] = label_encoders[col].transform(df[col])
    df = df[feature_cols]
    return scaler.transform(df)

def make_summary(data):
    amenity_score = sum(int(float(data.get(col, 0))) for col in AMENITY_COLS)
    amenities_yes = [col.replace('_', ' ').title()
                     for col in AMENITY_COLS if int(float(data.get(col, 0))) == 1]
    return {
        'location':      f"{data['sector']}, {data['district']}, {data['city']}",
        'total_area':    f"{data['total_area']} m2",
        'rooms':         int(data['rooms']),
        'bathrooms':     int(data['bathrooms']),
        'floor':         int(data['floor_number']),
        'year_built':    int(data['year_built']),
        'apartment_age': f"{2026 - int(data['year_built'])} years",
        'amenity_score': f"{amenity_score}/8",
        'amenities':     amenities_yes if amenities_yes else ['None']
    }

# ── Routes ───────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'project':    'Kigali Apartment Price Prediction',
        'author':     'Mukesha',
        'version':    '2.0.0',
        'best_model': MODEL_NAME,
        'status':     'API is running ✅',
        'endpoints': {
            'GET  /':              'API info',
            'GET  /health':        'Health check',
            'GET  /model':         'Model details',
            'POST /predict':       'Predict rent for one apartment',
            'POST /predict/batch': 'Predict rent for multiple apartments'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':     'healthy ✅',
        'model':      MODEL_NAME,
        'model_file': best_file,
        'models_dir': MODELS_DIR,
    })

@app.route('/model', methods=['GET'])
def model_info():
    return jsonify({
        'model_name':       MODEL_NAME,
        'model_file':       best_file,
        'features':         feature_cols,
        'total_features':   len(feature_cols),
        'valid_districts':  VALID_DISTRICTS,
        'valid_sectors':    VALID_SECTORS,
        'model_r2':         94.7,
        'ml_techniques': [
            '1. Amenity Score Aggregation',
            '2. Apartment Age Feature',
            '3. Area Per Room',
            '4. Total Rooms',
            '5. StandardScaler Normalization',
            '6. Label Encoding for categories',
            '7. GridSearchCV Hyperparameter Tuning',
            '8. 5-Fold Cross Validation',
            '9. Monotone Constraints (XGBoost)',
            '10. SelectKBest Feature Selection',
        ]
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        valid, error = validate_input(data)
        if not valid:
            return jsonify({'error': error}), 400

        X          = preprocess_input(data)
        prediction = best_model.predict(X)[0]
        rent       = round(float(prediction))

        # Confidence: for ensemble models use tree agreement (lower variance = higher confidence)
        if hasattr(best_model, 'estimators_'):
            tree_preds = np.array([t.predict(X)[0] for t in best_model.estimators_])
            cv = tree_preds.std() / tree_preds.mean() if tree_preds.mean() > 0 else 0
            confidence = round(float(max(60.0, min(99.9, (1 - cv) * 100))), 1)
        else:
            confidence = 94.7

        return jsonify({
            'status': 'success',
            'prediction': {
                'rent_price':           rent,
                'rent_price_formatted': f"{rent:,} RWF/month",
                'model_used':           MODEL_NAME,
                'confidence':           confidence,
                'confidence_label':     f"{confidence}%",
            },
            'apartment_summary': make_summary(data)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'apartments' not in data:
            return jsonify({'error': 'Provide JSON with "apartments" list'}), 400

        apartments = data['apartments']
        if not isinstance(apartments, list) or len(apartments) == 0:
            return jsonify({'error': '"apartments" must be a non-empty list'}), 400
        if len(apartments) > 50:
            return jsonify({'error': 'Maximum 50 apartments per request'}), 400

        results = []

        for i, apt in enumerate(apartments):
            valid, error = validate_input(apt)
            if not valid:
                results.append({'index': i, 'status': 'error', 'error': error})
                continue
            try:
                X    = preprocess_input(apt)
                rent = round(float(best_model.predict(X)[0]))
                results.append({
                    'index':  i,
                    'status': 'success',
                    'prediction': {
                        'rent_price':           rent,
                        'rent_price_formatted': f"{rent:,} RWF/month"
                    },
                    'apartment_summary': make_summary(apt)
                })
            except Exception as exc:
                results.append({'index': i, 'status': 'error', 'error': str(exc)})

        rents = [r['prediction']['rent_price'] for r in results if r['status'] == 'success']
        return jsonify({
            'status':     'success',
            'total':      len(apartments),
            'successful': len(rents),
            'model_used': MODEL_NAME,
            'summary': {
                'min_rent': f"{min(rents):,} RWF/month" if rents else 'N/A',
                'max_rent': f"{max(rents):,} RWF/month" if rents else 'N/A',
                'avg_rent': f"{round(sum(rents)/len(rents)):,} RWF/month" if rents else 'N/A',
            },
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found', 'status': 404}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Method not allowed', 'status': 405}), 405

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  Kigali Apartment Price Prediction API v2.0")
    print("  Running at: http://127.0.0.1:5000")
    print(f"  Model: {MODEL_NAME}")
    print("=" * 55 + "\n")
    port = int(os.environ.get('PORT', 5000))
app.run(debug=False, host='0.0.0.0', port=port)
