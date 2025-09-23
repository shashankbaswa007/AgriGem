import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain and Gemini Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# --- 1. INITIALIZE FLASK APP AND GLOBAL AGENT ---
app = Flask(__name__)
CORS(app)

# Global variable to hold the loaded agent
agent_executor = None

# --- 2. CROP PARAMETERS DATABASE FOR THE RECOMMENDATION ENGINE ---
CROP_PARAMETERS = {
    'rice': { 'water_per_hectare_liters': 12_000_000, 'nitrogen_per_tonne': 20.0 },
    'maize': { 'water_per_hectare_liters': 6_000_000, 'nitrogen_per_tonne': 22.0 },
    # Add other crops here...
    'default': { 'water_per_hectare_liters': 5_500_000, 'nitrogen_per_tonne': 15.0 }
}

# --- 3. RECOMMENDATION ENGINE LOGIC ---
def generate_recommendations(crop, area, target_yield):
    crop_key = crop.lower()
    params = CROP_PARAMETERS.get(crop_key, CROP_PARAMETERS['default'])

    total_water = params['water_per_hectare_liters'] * area
    total_nitrogen = target_yield * area * params['nitrogen_per_tonne']
    pesticide_recommendation = "Follow Integrated Pest Management (IPM) practices. Apply as needed based on scouting."

    report = f"""=== Crop Optimization Report for {crop.title()} ===

- Use {total_water:,.0f} liters of water
- Apply {total_nitrogen:.2f} kg of nitrogen fertilizer
- {pesticide_recommendation}

🌱 This plan is optimized for an estimated yield of {target_yield:.2f} tonnes per hectare."""

    if total_water > 10_000_000:
        report += "\n💧 Note: Water usage is high. Consider irrigating during the early morning or evening to reduce evaporation."
    return report

# --- 4. SETUP FUNCTION TO CREATE THE WEB SEARCH AGENT ---
def initialize_agent():
    global agent_executor
    print("--- Initializing Web Search Agent ---")
    load_dotenv()
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        tools = [TavilySearchResults(max_results=3)]
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        print("✅ Web Search Agent initialized successfully.")
    except Exception as e:
        print(f"❌ ERROR initializing agent: {e}")

# --- 5. CREATE THE API ENDPOINTS ---

# Endpoint for the Web Search Agent
@app.route('/ask', methods=['POST'])
def ask_gem():
    if not agent_executor:
        return jsonify({"error": "Agent is not initialized."}), 503
    question = request.json.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        response = agent_executor.invoke({"input": question})
        return jsonify({"answer": response['output']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint for the Recommendation Engine
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        crop = data.get('crop')
        area = float(data.get('area'))
        target_yield = float(data.get('target_yield'))
        if not all([crop, area, target_yield]):
            return jsonify({"error": "Missing required fields"}), 400
        report_text = generate_recommendations(crop, area, target_yield)
        return jsonify({"report": report_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 6. RUN THE SERVER ---
if __name__ == '__main__':
    initialize_agent()
    app.run(port=5001, debug=True)