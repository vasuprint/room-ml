# Roomfastwork – ระบบแนะนำห้องประชุมสำหรับงานราชการ

## 🚀 Quick Start - เริ่มใช้งานแบบรวดเร็ว

```bash
# 1. ติดตั้ง UV (ครั้งแรกเท่านั้น)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. ติดตั้ง Dependencies
uv sync

# 3. Train Model (ถ้ายังไม่มี)
uv run python code/train_room_model.py

# 4. รัน API Server
uv run uvicorn code.service.app:app --port 9000 --host 0.0.0.0

# 5. ทดสอบ API (เปิด terminal ใหม่)
# ⭐ ใช้ /recommend เพื่อได้ 3 ห้องพร้อมรายละเอียด
curl -X POST http://localhost:9000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "department": "งานบริหารทั่วไป",
    "duration_hours": 2.0,
    "event_period": "Morning",
    "seats": 30
  }'
```

✅ API จะทำงานที่ http://localhost:9000
📖 ดู API Docs ที่ http://localhost:9000/docs
⭐ **แนะนำ:** ใช้ `/recommend` endpoint เพื่อได้ผลลัพธ์ที่ครบถ้วน

## แนวคิดหลักของโครงการ

- ใช้ข้อมูลการจองห้องประชุมในอดีต (แผนก, ระยะเวลา, ช่วงเวลา, จำนวนที่นั่ง) เพื่อทำนายห้องที่เหมาะสมที่สุดให้ผู้ใช้ใหม่
- ประมวลผลและเทรนโมเดลด้วย Python/Scikit-learn ในสภาพแวดล้อมเชิงวิเคราะห์ (Jupyter Notebook หรือสคริปต์)
- แยกส่วนให้บริการโมเดลผ่าน API ด้วย FastAPI เพื่อให้ทีม Frontend (Vue) เรียกใช้งานได้อย่างอิสระ
- จัดเก็บโมเดลที่เทรนแล้วในรูปแบบไฟล์ `joblib` เพื่อให้โหลดกลับมาใช้งานได้รวดเร็วและสม่ำเสมอ

## โครงสร้างโปรเจกต์

```
clean/                     ชุดข้อมูล Excel ที่ผ่านการทำความสะอาดแล้ว (read-only)
code/                      โน้ตบุ๊ก, โมดูล Python, และบริการ API
  ├─ ClassificationRoom.ipynb   โน้ตบุ๊กหลักสำหรับทดลองโมเดล
  ├─ train_room_model.py        สคริปต์เทรนและบันทึกโมเดล
  ├─ models/                    เก็บไฟล์โมเดลที่เทรนแล้ว (.joblib)
  └─ service/app.py             FastAPI service เปิด endpoint /predict
pyproject.toml              รายชื่อ dependencies และการตั้งค่า uv
README.md                   เอกสารนี้
```

## การเตรียมสภาพแวดล้อมด้วย uv

1. ติดตั้ง uv (ครั้งเดียวต่อเครื่อง):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. ซิงก์ dependencies ตาม `pyproject.toml` (จะสร้าง virtualenv อัตโนมัติ):
   ```bash
   uv sync
   ```
3. หากต้องการเครื่องมือเสริม (black, ruff ฯลฯ):
   ```bash
   uv sync --group dev
   ```

## กระบวนการเทรนโมเดล

1. สร้างไฟล์โมเดลล่าสุดจากข้อมูลจริง:
   ```bash
   uv run python code/train_room_model.py
   ```
2. สคริปต์จะโหลดข้อมูลจาก `clean/cleaned_all_rooms.xlsx`, คัดกรองคลาสที่มีตัวอย่างน้อย, เทรน `DecisionTreeClassifier`
   พร้อม pipeline แปลงข้อมูล, พิมพ์ค่า accuracy/test report, แล้วบันทึกโมเดลที่ `code/models/room_classifier.joblib`
3. หากต้องการสำรวจหรือปรับแต่งโมเดลแบบโต้ตอบ ให้เปิดโน้ตบุ๊กด้วย `uv run jupyter lab`

## การให้บริการโมเดลผ่าน API

1. รันเซิร์ฟเวอร์ระหว่างพัฒนา:
   ```bash
   uv run uvicorn code.service.app:app --reload --port 9000
   ```
2. Endpoint สำคัญ:
    - `GET /health` ตรวจสอบสถานะเซิร์ฟเวอร์
    - `POST /predict` รับ JSON ฟีเจอร์และส่งคืนชื่อห้องที่โมเดลแนะนำ (1 ห้อง เฉพาะชื่อ)
    - `POST /predict_proba` (ถ้าโมเดลรองรับ) สำหรับดูความน่าจะเป็นต่อห้อง
    - `POST /recommend` ⭐ **[แนะนำให้ใช้ endpoint นี้]** รับ JSON ฟีเจอร์และส่งคืน 3
      อันดับห้องที่แนะนำพร้อมรายละเอียดครบถ้วน (UUID, ชื่อห้อง, สถานที่, ความจุ, ราคา)
3. ตัวอย่างคำสั่งทดสอบด้วย `httpie` (ติดตั้งด้วย `uv add httpie`):
   ```bash
   uv run http POST http://127.0.0.1:9000/predict \
     department=งานบริหารทั่วไป \
     duration_hours:=2.0 \
     event_period=บ่าย \
     seats:=35
   ```
4. สามารถเปิดเอกสารแบบโต้ตอบได้ที่ `http://127.0.0.1:9000/docs`

## การใช้งานด้วย Docker

### ติดตั้ง Docker

1. ติดตั้ง Docker Desktop: https://www.docker.com/products/docker-desktop
2. ตรวจสอบการติดตั้ง:
   ```bash
   docker --version
   docker-compose --version
   ```

### วิธีที่ 1: ใช้ Docker Compose

```bash
# 1.1 Train โมเดลก่อน (ถ้ายังไม่มีไฟล์โมเดล)
uv run python code/train_room_model.py

# 1.2 create an external network 
docker network create common_backend_network

# 1.3 check external network 
docker network ls

# 2. Build และรัน service
docker-compose up --build

# หรือรันแบบ background
docker-compose up -d --build

# 3. ตรวจสอบสถานะ
docker-compose ps

# 4. ดู logs
docker-compose logs -f

# 5. หยุด service
docker-compose down
```

API จะพร้อมใช้งานที่ `http://localhost:9000`

### วิธีที่ 2: ใช้ผ่าน Docker

```bash
# Build image
docker build -t roomfastwork:latest .

# Run container
docker run -d \
  --name roomfastwork-api \
  -p 9000:9000 \
  -v $(pwd)/code:/app/code \
  -v $(pwd)/code/models:/app/code/models \
  -v $(pwd)/clean:/app/clean:ro \
  roomfastwork:latest

# ตรวจสอบสถานะ
docker ps

# ดู logs
docker logs -f roomfastwork-api

# หยุด container
docker stop roomfastwork-api

# ลบ container
docker rm roomfastwork-api
```

### คำสั่ง Docker ที่มีประโยชน์

```bash
# เข้าไปใน container (debug)
docker-compose exec room-api bash

# Restart service
docker-compose restart

# ดูการใช้ resources
docker stats roomfastwork-api

# ลบ everything และเริ่มใหม่
docker-compose down -v
docker-compose up --build
```

### ทดสอบ API ผ่าน Docker

```bash
# Health check
curl http://localhost:9000/health

# Predict room
curl -X POST http://localhost:9000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "department": "งานบริหารทั่วไป",
    "duration_hours": 2.0,
    "event_period": "บ่าย",
    "seats": 35
  }'
```

### Production Deployment

สำหรับ production ให้:

1. แก้ไข `Dockerfile` - uncomment บรรทัด train model ถ้าต้องการ train ใน container
2. ตั้งค่า environment variables ใน `docker-compose.yml`:
   ```yaml
   environment:
     - LOG_LEVEL=warning
     - WORKERS=4
   ```
3. ใช้ reverse proxy (Nginx/Traefik) สำหรับ HTTPS
4. ตั้งค่า resource limits:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: "2"
         memory: 2G
   ```

## วิธีการใช้งาน API จริง

1. **API รับข้อมูลผ่าน HTTP POST body ในรูปแบบ JSON**

    - ส่งข้อมูลเป็น JSON object ใน body ของ HTTP request
    - Content-Type ต้องเป็น `application/json`

2. **ต้องเปิด FastAPI service ทิ้งไว้รันตลอดเวลา**

   ```bash
   uv run uvicorn code.service.app:app --reload --port 9000
   ```

    - คำสั่งนี้จะเปิด web server ที่ port 8000
    - ให้ terminal นี้รันอยู่ตลอด (ไม่ปิด)
    - API จะพร้อมรับ request ที่ `http://127.0.0.1:9000`

3. **คำสั่ง `uv run http POST ...` เป็นเพียงเครื่องมือทดสอบ**
    - HTTPie เป็นเครื่องมือ CLI สำหรับทดสอบ API
    - ในการใช้งานจริง คุณจะเรียก API จากโค้ดของคุณเอง

### โครงสร้างข้อมูลที่ส่งเข้า API

> ⚠️ **หมายเหตุสำคัญ:** แนะนำให้ใช้ `/recommend` endpoint แทน `/predict` เพราะจะได้ข้อมูลครบถ้วนกว่า (ได้ 3
> ห้องพร้อมรายละเอียด แทนที่จะได้แค่ชื่อห้องเดียว)

#### 1. `/recommend` endpoint ⭐ (แนะนำให้ใช้ - ให้ข้อมูลครบถ้วน)

**Input:**

```json
{
  "department": "งานบริหารทั่วไป",
  "duration_hours": 2.0,
  "event_period": "Morning",
  "seats": 30
}
```

**Output Structure:**

```json
{
  "recommendations": [
    {
      "rank": 1,
      // ลำดับที่แนะนำ (1 = แนะนำมากที่สุด)
      "room": {
        "id": "ed00f84a-...",
        // UUID สำหรับระบบจองห้อง
        "name": "ห้องประชุม อยุธยา – อาเซียน",
        // ชื่อห้องประชุม
        "location": "อาคารสำนักงานอธิการบดี",
        // สถานที่ตั้ง
        "capacity_min": 15,
        // จำนวนคนขั้นต่ำ
        "capacity_max": 30,
        // จำนวนคนสูงสุด
        "price": 1000
        // ราคา (บาท/ชั่วโมง)
      }
    },
    {
      "rank": 2,
      // ลำดับที่ 2
      "room": {
        "id": "1ad74b1c-...",
        "name": "ห้องประชุมมหาวิทยาลัย",
        "location": "มหาวิทยาลัยราชภัฏพระนครศรีอยุธยา",
        "capacity_min": 200,
        "capacity_max": 700,
        "price": 2500
      }
    },
    {
      "rank": 3,
      // ลำดับที่ 3
      "room": {
        "id": "38f4c25b-...",
        "name": "หอประชุม 1",
        "location": "อาคารป่าตอง",
        "capacity_min": 12,
        "capacity_max": 20,
        "price": 400
      }
    }
  ],
  "request_id": "req-17da3ade"
  // Request ID สำหรับ tracking
}
```

#### 2. `/predict` endpoint (พื้นฐาน - ให้แค่ชื่อห้อง)

> ⚠️ **ไม่แนะนำ:** Endpoint นี้ให้ข้อมูลจำกัด ควรใช้ `/recommend` แทน

**Input:**

```json
{
  "department": "งานบริหารทั่วไป",
  "duration_hours": 2.0,
  "event_period": "บ่าย",
  "seats": 35
}
```

**Output:**

```json
{
  "room": "ห้องประชุมใหญ่"
  // ได้แค่ชื่อห้องเดียว
}
```

#### 📊 เปรียบเทียบ Endpoints

| Feature           | `/predict`       | `/recommend` ⭐ |
|-------------------|------------------|----------------|
| จำนวนห้องที่แนะนำ | 1 ห้อง           | 3 ห้อง         |
| ข้อมูลที่ได้      | ชื่อห้องเท่านั้น | ข้อมูลครบถ้วน  |
| UUID              | ❌                | ✅              |
| สถานที่           | ❌                | ✅              |
| ความจุ            | ❌                | ✅              |
| ราคา              | ❌                | ✅              |
| การจัดอันดับ      | ❌                | ✅              |
| **แนะนำให้ใช้**   | ❌                | ✅ ใช้อันนี้    |

**คำอธิบายแต่ละฟีลด์:**

- `department` (string): ชื่อหน่วยงาน/แผนกที่จอง
- `duration_hours` (number): ระยะเวลาการใช้งาน (ชั่วโมง) ต้องมากกว่าหรือเท่ากับ 0
- `event_period` (string): ช่วงเวลา เช่น `"Morning"`, `"Afternoon"`, `"All Day"`
- `seats` (integer): จำนวนที่นั่งที่ต้องการ ต้องมากกว่าหรือเท่ากับ 1

### ตัวอย่างการใช้งานในภาษาต่างๆ

#### A. JavaScript / Vue.js (Frontend)

```javascript
// ใช้ fetch API กับ /recommend endpoint (แนะนำ - ได้ข้อมูลครบถ้วน)
async function getRecommendations() {
    const response = await fetch("http://127.0.0.1:9000/recommend", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            department: "งานบริหารทั่วไป",
            duration_hours: 2.0,
            event_period: "Morning",
            seats: 30,
        }),
    });

    const data = await response.json();

    // แสดงผล 3 อันดับห้องที่แนะนำ
    data.recommendations.forEach(rec => {
        console.log(`อันดับ ${rec.rank}: ${rec.room.name}`);
        console.log(`  - สถานที่: ${rec.room.location}`);
        console.log(`  - ความจุ: ${rec.room.capacity_min}-${rec.room.capacity_max} คน`);
        console.log(`  - ราคา: ${rec.room.price} บาท/ชั่วโมง`);
    });

    return data.recommendations;
}

// หรือใช้ axios (ติดตั้งด้วย npm install axios)
import axios from "axios";

async function getRecommendationsWithAxios() {
    const response = await axios.post("http://127.0.0.1:9000/recommend", {
        department: "งานบริหารทั่วไป",
        duration_hours: 2.0,
        event_period: "Morning",
        seats: 30,
    });

    return response.data.recommendations;
}
```

#### B. PHP / Laravel (Backend)

```php
use Illuminate\Support\Facades\Http;

// ใช้ Laravel HTTP Client
$response = Http::post('http://127.0.0.1:9000/predict', [
    'department' => 'งานบริหารทั่วไป',
    'duration_hours' => 2.0,
    'event_period' => 'บ่าย',
    'seats' => 35
]);

if ($response->successful()) {
    $predictedRoom = $response->json()['predicted_room'];
    echo "ห้องที่แนะนำ: " . $predictedRoom;
}
```

#### C. Python (Backend Script)

```python
import requests

response = requests.post('http://127.0.0.1:9000/predict', json={
    'department': 'งานบริหารทั่วไป',
    'duration_hours': 2.0,
    'event_period': 'บ่าย',
    'seats': 35
})

if response.status_code == 200:
    data = response.json()
    print(f"ห้องที่แนะนำ: {data['predicted_room']}")
```

#### D. HTML + JavaScript (Simple Web Page)

```html
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8"/>
    <title>ระบบแนะนำห้องประชุม</title>
</head>
<body>
<h1>ระบบแนะนำห้องประชุม</h1>

<form id="roomForm">
    <div>
        <label>หน่วยงาน:</label>
        <input type="text" name="department" required/>
    </div>

    <div>
        <label>จำนวนชั่วโมง:</label>
        <input
                type="number"
                name="duration_hours"
                step="0.5"
                min="0"
                required
        />
    </div>

    <div>
        <label>ช่วงเวลา:</label>
        <select name="event_period" required>
            <option value="เช้า">เช้า</option>
            <option value="บ่าย">บ่าย</option>
            <option value="เย็น">เย็น</option>
        </select>
    </div>

    <div>
        <label>จำนวนที่นั่ง:</label>
        <input type="number" name="seats" min="1" required/>
    </div>

    <button type="submit">ค้นหาห้องประชุม</button>
</form>

<div id="result"></div>

<script>
    document
            .getElementById("roomForm")
            .addEventListener("submit", async (e) => {
                e.preventDefault();

                // รวบรวมข้อมูลจากฟอร์ม
                const formData = new FormData(e.target);
                const data = {
                    department: formData.get("department"),
                    duration_hours: parseFloat(formData.get("duration_hours")),
                    event_period: formData.get("event_period"),
                    seats: parseInt(formData.get("seats")),
                };

                try {
                    // เรียก API
                    const response = await fetch("http://127.0.0.1:9000/predict", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify(data),
                    });

                    const result = await response.json();

                    // แสดงผลลัพธ์
                    document.getElementById(
                            "result"
                    ).innerHTML = `<h2>ห้องที่แนะนำ: ${result.predicted_room}</h2>`;
                } catch (error) {
                    document.getElementById(
                            "result"
                    ).innerHTML = `<p style="color: red;">เกิดข้อผิดพลาด: ${error.message}</p>`;
                }
            });
</script>
</body>
</html>
```

### API Endpoint อื่นๆ

#### 1. Health Check

```bash
GET http://127.0.0.1:9000/health
```

ตรวจสอบว่า API service รันอยู่หรือไม่

#### 2. Predict with Probability

```bash
POST http://127.0.0.1:9000/predict_proba
Content-Type: application/json

{
  "department": "งานบริหารทั่วไป",
  "duration_hours": 2.0,
  "event_period": "บ่าย",
  "seats": 35
}
```

ได้ผลลัพธ์แบบมี probability ของแต่ละห้อง:

```json
{
  "predicted_room": "ห้องประชุมใหญ่",
  "probabilities": {
    "ห้องประชุมใหญ่": 0.75,
    "ห้องประชุมกลาง": 0.15,
    "ห้องประชุมเล็ก": 0.1
  }
}
```

### ขั้นตอนการใช้งานจริงทั้งหมด

```
┌─────────────────────────────────────────┐
│ 1. เริ่ม FastAPI Service (Terminal 1)  │
│    $ uv run uvicorn code.service.app:app│
│      --reload --port 8000               │
│    ► Service รันที่ http://127.0.0.1:9000│
│    ► เปิดทิ้งไว้ตลอด                    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 2. เรียกใช้งาน API จากโค้ดของคุณ      │
│    - เขียน Frontend (Vue/React/HTML)    │
│    - เขียน Backend (Laravel/Django)     │
│    - เขียน Script (Python/Node.js)      │
│    - ส่งข้อมูลผ่าน HTTP POST           │
│    - Body เป็น JSON                     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│ 3. API ประมวลผลและตอบกลับ              │
│    ► โมเดล ML ทำนายห้องที่เหมาะสม      │
│    ► ส่งผลลัพธ์เป็น JSON               │
└─────────────────────────────────────────┘
```

### หมายเหตุสำคัญ

- **CORS**: FastAPI เปิด CORS สำหรับ `http://localhost:5173` (Vue dev server) แล้ว

    - หากใช้ port อื่น ให้แก้ไขใน `code/service/app.py` ที่บรรทัด `allow_origins`
    - สำหรับ production ให้เปลี่ยนเป็น domain จริงของคุณ

- **Error Handling**: ควรจัดการ error ที่อาจเกิดขึ้น เช่น:

    - Network error (API ไม่ตอบสนอง)
    - Validation error (ข้อมูลไม่ถูกต้อง)
    - Server error (API เกิดข้อผิดพลาด)

- **Production Deployment**:
    - อย่าใช้ `--reload` ใน production
    - ใช้ reverse proxy (Nginx, Apache) หน้า FastAPI
    - ตั้งค่า HTTPS และ security headers

## การเชื่อมต่อกับ Vue

1. ตั้งค่า `.env.local` ในโปรเจกต์ Vue:
   ```
   VITE_API_BASE_URL=http://127.0.0.1:9000
   ```
2. สร้าง service client (ตัวอย่างใช้ Axios):

   ```ts
   // src/services/roomService.ts
   import axios from "axios";

   const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL });

   export async function predictRoom(payload) {
     const { data } = await api.post("/predict", payload);
     return data.room;
   }
   ```

3. เรียกใช้ใน component และแสดงผลห้องที่โมเดลทำนาย
4. FastAPI เปิด CORS ไว้สำหรับ `http://localhost:5173` แล้ว หาก deploy จริงให้ปรับ `allow_origins`
   ให้ตรงกับโดเมนที่ใช้งาน

## การทดสอบและตรวจสอบผลลัพธ์

- **ชุดทดสอบอัตโนมัติ**: เพิ่มไฟล์ลงใน `code/tests/` แล้วรัน `uv run pytest -q`
- **ตรวจสอบโมเดล**: หลังเทรนสามารถเปิดไฟล์ `code/models/room_classifier.joblib` ด้วยโน้ตบุ๊กเพื่อตรวจผล เช่น confusion
  matrix
- **ตรวจสอบ API**: ใช้ `curl` หรือ Swagger UI เพื่อยืนยันการตอบกลับเมื่อกรอกข้อมูลหลากหลายกรณี (เช่น
  จำนวนที่นั่งสูง/ต่ำ, แผนกหายาก)
- **การตรวจสอบรวมกับ Vue**: เมื่อเปิด front-end (`npm run dev`) ให้ทดสอบฟอร์มจริงว่ารับคำตอบตรงกับผลจาก `httpie`
