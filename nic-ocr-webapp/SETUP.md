# Setup Guide

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | [python.org](https://python.org) |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) |
| MySQL | 8.x | Any 8.x release |
| Tesseract OCR | 5.x | Must include Sinhala (`sin`) language pack |

---

## 1. Install Tesseract OCR

Tesseract must be installed **system-wide** with the Sinhala language pack.

### Windows

Download and run the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).  
During installation, select **Additional language data → Sinhala**.

Default install path: `C:/Program Files/Tesseract-OCR/`

### macOS

```bash
brew install tesseract
brew install tesseract-lang   # includes all language packs (sin, etc.)
```

Find your paths after install:
```bash
which tesseract          # binary path
brew --prefix           # use this to find tessdata: <prefix>/share/tessdata
```

### Linux (Ubuntu/Debian)

```bash
sudo apt install tesseract-ocr tesseract-ocr-sin
which tesseract          # usually /usr/bin/tesseract
# tessdata is usually at /usr/share/tesseract-ocr/5/tessdata
```

---

## 2. Clone the Repository

```bash
git clone <repo-url>
cd OCR_Project/nic-ocr-webapp
```

---

## 3. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your local values:

**Windows:**
```env
TESSDATA_PREFIX=C:/Program Files/Tesseract-OCR/tessdata
TESSERACT_PATH=C:/Program Files/Tesseract-OCR/tesseract.exe
```

**macOS (Apple Silicon):**
```env
TESSDATA_PREFIX=/opt/homebrew/share/tessdata
TESSERACT_PATH=/opt/homebrew/bin/tesseract
```

**macOS (Intel):**
```env
TESSDATA_PREFIX=/usr/local/share/tessdata
TESSERACT_PATH=/usr/local/bin/tesseract
```

**Linux:**
```env
TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata
TESSERACT_PATH=/usr/bin/tesseract
```

Fill in the remaining values:
```env
BASE_SIN_MODEL=sin
STORAGE_PATH=./storage

DB_HOST=localhost
DB_PORT=3306
DB_NAME=nic_ocr
DB_USER=root
DB_PASSWORD=your_password_here

CORS_ORIGINS=http://localhost:5173
```

### Create the database

```sql
CREATE DATABASE nic_ocr;
```

### Start the backend

```bash
uvicorn app.main:app --reload
```

Runs on `http://localhost:8000`. Tables and storage directories are created automatically on first run.

---

## 4. Frontend Setup

```bash
cd ../frontend

npm install
npm run dev
```

Runs on `http://localhost:5173`.

---

## Verify Everything Works

| Check | URL |
|-------|-----|
| Backend health | `http://localhost:8000/health` → `{"status": "ok"}` |
| API docs | `http://localhost:8000/docs` |
| Frontend | `http://localhost:5173` |

---

## Common Issues

**Tesseract not found** — Make sure `TESSERACT_PATH` in `.env` points to the actual binary. Run `which tesseract` (macOS/Linux) or `where tesseract` (Windows) to confirm.

**Sinhala language missing** — Run `tesseract --list-langs` and check that `sin` appears. If not, reinstall with the language pack.

**Database connection error** — Make sure MySQL is running and the `nic_ocr` database exists before starting the backend.

**`.env` not found** — The app will not start without a `.env` file. Copy `.env.example` and fill in your values.

**`networkx` / package install fails on Python 3.10** — Several dependencies require Python 3.11+. Upgrade via [python.org](https://python.org) or `pyenv install 3.11`.
