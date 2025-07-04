# HealthCareAssistant

A conversational AI assistant for healthcare data, built with Flask, LangChain, OpenAI, and FAISS vector search.

## Features
- Chat interface for healthcare Q&A
- Uses OpenAI GPT-3.5-turbo for language understanding
- Embedding and retrieval of patient and allergy data using FAISS
- Modular knowledge base for different healthcare domains
- Easily extensible with new tools and data

## Project Structure
```
HealthCareAssistant/
├── backend/
│   └── app.py                # Main Flask backend and agent logic
├── allergies_faiss_index/    # FAISS index for allergy data
├── patients_faiss_index/     # FAISS index for patient data
├── transformed_data/         # (Ignored) Large data files
├── myenv/                    # (Ignored) Python virtual environment
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
```

## Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/RISHIKANTH-S/EHR-Assistant.git
   cd EHR-Assistant
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv myenv
   myenv\Scripts\activate  # On Windows
   # or
   source myenv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key:**
   - Create a `.env` file in the root directory:
     ```env
     OPENAI_API_KEY=your_openai_api_key_here
     ```

5. **Run the Flask app:**
   ```sh
   cd backend
   python app.py
   ```
   The app will be available at `http://127.0.0.1:5000/`

## Usage
- Open the chat interface in your browser.
- Ask questions about patient details or allergies.
- The assistant will use the appropriate knowledge base and tools to answer.

## Notes
- Large files and indexes are ignored by git (see `.gitignore`).
- If you need to rebuild FAISS indexes, use your data and the appropriate LangChain scripts.
- For production, consider using a production-ready server (e.g., Gunicorn, Uvicorn) and securing your API keys.
- Provided Datasets are broad, we only worked with patient, allergies knowledge bases. we provided data about medications,immunizations etc. Interested people can create tools on each knowledge base

## License
MIT License
