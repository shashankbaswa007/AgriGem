import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain and Gemini Imports
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- 1. INITIALIZE FLASK APP AND GLOBAL VARIABLES ---
app = Flask(__name__)
CORS(app) # Allows your frontend to make requests

# Global variable to hold the loaded RAG chain
rag_chain = None

# --- 2. RECOMMENDATION ENGINE LOGIC ---
def generate_recommendations(crop, area, target_yield):
    """
    Generates a crop optimization report based on simple agricultural rules.
    """
    # These are example rules. You can find more precise numbers in agricultural guides.
    
    # Rule 1: Water Requirement
    if crop.lower() == 'rice':
        water_per_hectare_liters = 12_000_000  # Approx. 12 million liters
    else: # Default for other crops
        water_per_hectare_liters = 6_000_000
    total_water = water_per_hectare_liters * area

    # Rule 2: Nitrogen Requirement
    if crop.lower() == 'rice':
        nitrogen_per_tonne = 20.0 # Approx. 20 kg of N per tonne of rice grain
    else:
        nitrogen_per_tonne = 15.0
    total_nitrogen = target_yield * area * nitrogen_per_tonne

    # Rule 3: Pesticide is a general recommendation
    pesticide_recommendation = "Follow Integrated Pest Management (IPM) practices. Apply as needed based on scouting."

    # --- Format the report using an f-string ---
    report = f"""=== Crop Optimization Report ===

- Use {total_water:,.0f} liters of water
- Apply {total_nitrogen:.2f} kg of nitrogen fertilizer
- {pesticide_recommendation}

🌱 This plan is optimized for an estimated yield of {target_yield:.2f} tonnes per hectare.
"""

    # Conditional note for high water usage
    if total_water > 10_000_000:
        report += "\n💧 Note: Water usage is high. Consider irrigating during the early morning or evening to reduce evaporation."
        
    return report


# --- 3. SETUP FUNCTION TO LOAD KNOWLEDGE BASE ON STARTUP ---
def initialize_chatbot():
    """
    Loads all documents and builds the RAG chain for the chatbot.
    """
    global rag_chain
    
    print("--- Initializing Gemini Gem Chatbot ---")
    load_dotenv()

    try:
        print("\nLoading knowledge base for the Gem...")
        excel_loader = UnstructuredExcelLoader("knowledge_base/odisha_stats.xlsx", mode="single")
        documents = excel_loader.load()
        print(f"Loaded {len(documents)} documents into the knowledge base.")

        print("Creating vector embeddings and storing in Chroma...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(documents, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print("✅ Vector store created successfully.")

        template = """
        You are a helpful assistant for farmers in Odisha. Answer based ONLY on the provided context.
        CONTEXT: {context}
        QUESTION: {question}
        ANSWER:
        """
        prompt = PromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("✅ Gemini Gem (RAG Chain) initialized successfully.")
    except Exception as e:
        print(f"❌ ERROR initializing Gem: {e}")


# --- 4. CREATE THE API ENDPOINTS ---

# --- Endpoint for the Gemini Gem Chatbot ---
@app.route('/ask', methods=['POST'])
def ask_gem():
    if not rag_chain:
        return jsonify({"error": "Chatbot is not initialized. Check server logs."}), 503

    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
        
    try:
        answer = rag_chain.invoke(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Endpoint for the Recommendation Engine ---
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        crop = data.get('crop')
        area = float(data.get('area'))
        target_yield = float(data.get('target_yield'))

        if not all([crop, area, target_yield]):
            return jsonify({"error": "Missing required fields: crop, area, target_yield"}), 400
            
        report_text = generate_recommendations(crop, area, target_yield)
        
        return jsonify({"report": report_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- 5. RUN THE SERVER ---
if __name__ == '__main__':
    initialize_chatbot()
    app.run(port=5001, debug=True)