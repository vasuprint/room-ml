"""Check what features the trained model expects"""
import joblib
import pandas as pd

# Load the model
model = joblib.load('code/models/room_classifier.joblib')

print("="*50)
print("MODEL INSPECTION")
print("="*50)

# Check pipeline steps
print("\n1. Pipeline steps:")
for name, step in model.named_steps.items():
    print(f"   - {name}: {type(step).__name__}")

# Check preprocessor
if 'preprocess' in model.named_steps:
    preprocessor = model.named_steps['preprocess']
    print("\n2. Preprocessor transformers:")
    for name, transformer, columns in preprocessor.transformers_:
        print(f"   - {name}: {columns}")

# Test with sample data WITHOUT price
print("\n3. Testing without price:")
test_data_no_price = pd.DataFrame([{
    'department': 'งานบริหารทั่วไป',
    'duration_hours': 2.0,
    'event_period': 'Morning',
    'seats': 30
}])

try:
    prediction = model.predict(test_data_no_price)
    print(f"   ✅ SUCCESS - Predicted: {prediction[0]}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

# Test with sample data WITH price
print("\n4. Testing with price:")
test_data_with_price = pd.DataFrame([{
    'department': 'งานบริหารทั่วไป',
    'duration_hours': 2.0,
    'event_period': 'Morning',
    'seats': 30,
    'price': 1000.0
}])

try:
    prediction = model.predict(test_data_with_price)
    print(f"   ✅ SUCCESS - Predicted: {prediction[0]}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print("\n" + "="*50)