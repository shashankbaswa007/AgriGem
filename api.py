import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain & Gemini imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# ----------------------------
# 1. Initialize Flask
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# 2. Load environment variables
# ----------------------------
load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

if not gemini_api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in environment variables!")

# ----------------------------
# 3. Initialize LLM + Tools
# ----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
search_tool = TavilySearchResults(max_results=3)
tools = [search_tool]

# ----------------------------
# 4. Setup LangChain Agent
# ----------------------------
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------------------
# 5. Routes
# ----------------------------

@app.route("/ask", methods=["POST"])
def ask():
    """
    Ask a free-text question to the agent.
    """
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "No question provided"}), 400

        response = agent_executor.invoke({"input": question})
        return jsonify({"answer": response["output"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    Recommend crop plan based on inputs.
    """
    try:
        data = request.get_json()
        crop = data.get("crop", "unknown")
        area = float(data.get("area", 1))
        target_yield = float(data.get("target_yield", 3))

        # Simple fixed logic (replace with your ML/DB logic later)
        water_usage = 12000 * area * target_yield
        nitrogen = 40 * area * target_yield

        report = f"""=== Crop Optimization Report for {crop.capitalize()} ===

- Use {water_usage:,.0f} liters of water
- Apply {nitrogen:.2f} kg of nitrogen fertilizer
- Follow Integrated Pest Management (IPM) practices. Apply as needed based on scouting.

🌱 This plan is optimized for an estimated yield of {target_yield:.2f} tonnes per hectare.
💧 Note: Water usage is high. Consider irrigating during early morning/evening to reduce evaporation."""

        return jsonify({"report": report})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------
# 6. Main entry point
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
