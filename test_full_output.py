"""Test and show full output from /recommend endpoint"""
import requests
import json

# Test data
test_data = {
    "department": "งานบริหารทั่วไป",
    "duration_hours": 2.0,
    "event_period": "Morning",
    "seats": 30
}

print("="*70)
print("📍 TESTING /recommend ENDPOINT - FULL OUTPUT")
print("="*70)
print("\n📥 INPUT:")
print(json.dumps(test_data, indent=2, ensure_ascii=False))
print("\n" + "-"*70)

# Make request
response = requests.post("http://localhost:9000/recommend", json=test_data)

print("\n📤 OUTPUT:")
print(f"Status Code: {response.status_code}")
print("\nFull Response JSON:")
print(json.dumps(response.json(), indent=2, ensure_ascii=False))

print("\n" + "="*70)

# Also test with different scenarios
test_cases = [
    {
        "name": "Small Room (8 seats)",
        "data": {
            "department": "งานวิชาการ",
            "duration_hours": 1.5,
            "event_period": "Afternoon",
            "seats": 8
        }
    },
    {
        "name": "Large Hall (500 seats)",
        "data": {
            "department": "มหาวิทยาลัย",
            "duration_hours": 4.0,
            "event_period": "All Day",
            "seats": 500
        }
    }
]

print("\n📊 ADDITIONAL TEST CASES:")
print("="*70)

for test in test_cases:
    print(f"\n🔸 {test['name']}")
    print(f"Input: {json.dumps(test['data'], ensure_ascii=False)}")

    response = requests.post("http://localhost:9000/recommend", json=test['data'])

    if response.status_code == 200:
        print(f"Output:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    else:
        print(f"Error: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    print("-"*70)