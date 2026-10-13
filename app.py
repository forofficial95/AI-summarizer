
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

HEADERS = {
    "User-Agent": "AI-Summary-Student-Project/1.0"
}


def get_wikipedia_summary(topic):
    # Step 1: Find the best matching Wikipedia article
    response = requests.get(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srlimit": 1,
            "format": "json"
        },
        headers=HEADERS,
        timeout=15
    )
    response.raise_for_status()

    results = response.json()["query"]["search"]

    if not results:
        return None, None

    title = results[0]["title"]

    # Step 2: Retrieve the article's plain text
    response = requests.get(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "titles": title,
            "format": "json"
        },
        headers=HEADERS,
        timeout=15
    )
    response.raise_for_status()

    pages = response.json()["query"]["pages"]
    article = next(iter(pages.values()))
    text = article.get("extract", "")

    if not text.strip():
        return title, None

    return title, text


def generate_summary(title, article):
    # Limit the input to avoid sending an excessively large article
    prompt = f"""
You are a helpful educational assistant.

Summarize the following Wikipedia article about "{title}"
in simple, easy-to-understand English.

Include:
- What the topic is
- Important facts
- Why it is significant

Use short paragraphs and bullet points where useful.
Use only information supported by the supplied article.
If information is missing, do not invent it.

Wikipedia article:
{article[:12000]}
"""

    response = requests.post(
        OLLAMA_API,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()

    return response.json()["response"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    topic = data.get("topic", "").strip()

    if not topic:
        return jsonify({"error": "Please enter a topic."}), 400

    if len(topic) > 200:
        return jsonify({
            "error": "Please enter a topic under 200 characters."
        }), 400

    try:
        title, article = get_wikipedia_summary(topic)

        if title is None:
            return jsonify({
                "error": "No matching Wikipedia article was found."
            }), 404

        if not article:
            return jsonify({
                "error": "The Wikipedia article has no usable text."
            }), 404

        summary = generate_summary(title, article)

        return jsonify({
            "title": title,
            "source": "Wikipedia",
            "summary": summary
        })

    except requests.Timeout:
        return jsonify({
            "error": "The request timed out. Please try again."
        }), 504

    except requests.RequestException:
        app.logger.exception("Request to Wikipedia or Ollama failed")
        return jsonify({
            "error": (
                "Could not reach Wikipedia or Ollama. "
                "Check your internet connection and Ollama."
            )
        }), 502

    except (KeyError, ValueError, TypeError):
        app.logger.exception("Unexpected API response")
        return jsonify({
            "error": "The response could not be processed."
        }), 502


if __name__ == "__main__":
    app.run(debug=True)
