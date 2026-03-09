# 💄 Beauty Ingredient Recommender

An AI-powered beauty ingredient recommendation system built using **LangChain** and **Azure OpenAI**.

The system suggests cosmetic ingredients based on a user's skin type, skin needs, concerns, and desired product type. It generates structured ingredient reports in strict JSON format using Pydantic output parsing, making it suitable for integration with APIs, databases, and frontend applications.

---

## 🎯 Project Goals

This project demonstrates:

- Reusable LangChain prompt design
- Structured LLM output parsing
- Schema validation using Pydantic
- Integration with Azure OpenAI
- Reliable and machine-readable AI responses

---

## ✨ Features

- AI-based ingredient recommendations
- Supports skin and hair related concerns
- Structured JSON output
- Schema validation using `PydanticOutputParser`
- Reusable prompt templates
- Error handling for invalid LLM responses
- Modular project architecture

---

## 📥 Supported Inputs

| Parameter | Description | Example |
|-----------|-------------|---------|
| `skin_type` | Type of skin | `oily`, `dry`, `sensitive` |
| `skin_needs` | Skin characteristics | `oiliness`, `tightness`, `sensitivity` |
| `concern` | Skin or hair issue | `acne`, `pigmentation`, `hair fall` |
| `product_type` | Cosmetic product category | `serum`, `shampoo`, `cleanser` |

---

## 📤 Example Output

```json
{
  "text_summary": "Recommended ingredients for oily acne-prone skin.",
  "recommended_ingredients": [
    {
      "ingredient_name": "Niacinamide",
      "function": "Regulates sebum production and improves skin barrier",
      "recommended_products": ["serum", "moisturiser"],
      "usage_percentage": "2-5%",
      "safety_notes": "Generally safe for most skin types",
      "suitable_for_sensitive_skin": true
    }
  ],
  "avoid_ingredients": [
    {
      "ingredient_name": "Coconut Oil",
      "reason_to_avoid": "Highly comedogenic and may worsen acne"
    }
  ]
}
```

---

## 🧠 System Architecture

```
User Input
   │
   ▼
Input Validation
   │
   ▼
LangChain Prompt Template
   │
   ▼
Azure OpenAI (LLM)
   │
   ▼
PydanticOutputParser
   │
   ▼
Validated JSON Ingredient Report
```

---

## 🗂 Project Structure

```
beauty-recommender/
│
├── app/
│   ├── chain.py        # LangChain pipeline
│   ├── llm.py          # Azure OpenAI configuration
│   ├── prompt.py       # PromptTemplate definitions
│   └── schema.py       # Pydantic schemas
│
├── main.py             # Test runs
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/beauty-recommender.git
cd beauty-recommender
```

### 2. Create Virtual Environment

Using Conda:

```bash
conda create -n llm python=3.10
conda activate llm
```

Or using venv:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
ENDPOINT=https://your-azure-openai-endpoint
API_VERSION=2024-02-01
AZURE_OPENAI_API_KEY=your_api_key
DEPLOYMENT_NAME=your_model_deployment
```

---

## ▶️ Running the Application

```bash
python main.py
```

This will execute multiple test cases such as:

- oily skin + acne + serum
- dry skin + pigmentation + moisturiser
- sensitive scalp + hair fall + shampoo

---

## 🧩 Core Components

| Component | File | Description |
|-----------|------|-------------|
| `AzureChatOpenAI` | `app/llm.py` | Handles interaction with Azure OpenAI |
| `PromptTemplate` | `app/prompt.py` | Reusable prompt that dynamically injects user inputs |
| `PydanticOutputParser` | `app/schema.py` | Ensures strict JSON output matching the schema |
| `LangChain Chain` | `app/chain.py` | Combines Prompt → LLM → Output Parser |

---

## 🛡 Error Handling

The system handles:

- Invalid JSON outputs
- Schema mismatches
- Missing required fields

When parsing fails, the system returns a safe error response:

```json
{
  "ok": false,
  "error": "Invalid output format produced by LLM"
}
```

---

## 🚀 Future Improvements

- Dermatology knowledge base using RAG
- Ingredient database integration
- Cosmetic product recommendation system
- Personalized skincare routines
- Multi-agent beauty formulation system

---

## 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| LangChain | LLM orchestration framework |
| Azure OpenAI | LLM provider |
| Pydantic | Output schema validation |
| dotenv | Environment configuration |
| VS Code | Development IDE |
