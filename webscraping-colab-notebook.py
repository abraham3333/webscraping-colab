# Web Scraping with Open Source LLM - Google Colab Notebook

# Install required libraries
!pip install transformers requests beautifulsoup4 pandas

import os
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from transformers import pipeline

# Base Scraper
class BaseScraper:
    def fetch_content(self, url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def extract(self, content: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")

# HTML Scraper
class HTMLScraper(BaseScraper):
    def extract(self, content: str) -> Dict[str, Any]:
        soup = BeautifulSoup(content, 'html.parser')
        return {
            'title': soup.title.string if soup.title else '',
            'text': soup.get_text(),
            'links': [a['href'] for a in soup.find_all('a', href=True)],
        }

# JSON Scraper
class JSONScraper(BaseScraper):
    def extract(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON content"}

# Web Extractor
class WebExtractor:
    def __init__(self):
        self.html_scraper = HTMLScraper()
        self.json_scraper = JSONScraper()
        self.model = pipeline("text-generation", model="gpt2")  # Using GPT-2 as an example
        self.current_url = None
        self.current_content = None
        self.preprocessed_content = None

    def fetch_url(self, url: str) -> str:
        self.current_url = url
        self.current_content = self.html_scraper.fetch_content(self.current_url)
        self.preprocessed_content = self._preprocess_content(self.current_content)
        return f"I've fetched and preprocessed the content from {self.current_url}. What would you like to know about it?"

    def _preprocess_content(self, content: str) -> str:
        soup = BeautifulSoup(content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
        return '\n'.join(chunk for chunk in chunks if chunk)

    def process_query(self, user_input: str) -> str:
        if user_input.lower().startswith("http"):
            return self.fetch_url(user_input)
        elif not self.current_content:
            return "Please provide a URL first before asking for information."
        else:
            return self._extract_info(user_input)

    def _extract_info(self, query: str) -> str:
        if not self.preprocessed_content:
            return "Please provide a URL first before asking for information."

        prompt = f"""Based on the following webpage content and the user's request, extract the relevant information.
        Always present the data as a JSON array of objects.
        Webpage content: {self.preprocessed_content[:500]}...
        Human: {query}
        AI:"""

        response = self.model(prompt, max_length=500, num_return_sequences=1)[0]['generated_text']
        
        # Extract the JSON part from the response
        json_start = response.find('[')
        json_end = response.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = response[json_start:json_end]
            try:
                extracted_data = json.loads(json_str)
                return self._format_result(json.dumps(extracted_data), query)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON data. Raw data: {json_str[:500]}..."
        else:
            return "Error: Could not extract JSON data from the model's response."

    def _format_result(self, extracted_data: str, query: str) -> str:
        if 'json' in query.lower():
            return self._format_as_json(extracted_data)
        elif 'csv' in query.lower():
            return self._format_as_csv(extracted_data)
        else:
            return self._format_as_text(extracted_data)

    def _format_as_json(self, data: str) -> str:
        try:
            parsed_data = json.loads(data)
            return f"```json\n{json.dumps(parsed_data, indent=2)}\n```"
        except json.JSONDecodeError:
            return f"Error: Invalid JSON data. Raw data: {data[:500]}..."

    def _format_as_csv(self, data: str) -> str:
        try:
            parsed_data = json.loads(data)
            if not parsed_data:
                return "No data to convert to CSV."
            df = pd.DataFrame(parsed_data)
            return f"```csv\n{df.to_csv(index=False)}\n```"
        except json.JSONDecodeError:
            return f"Error: Invalid JSON data. Raw data: {data[:500]}..."

    def _format_as_text(self, data: str) -> str:
        try:
            parsed_data = json.loads(data)
            return "\n".join([", ".join([f"{k}: {v}" for k, v in item.items()]) for item in parsed_data])
        except json.JSONDecodeError:
            return data

# Usage example
extractor = WebExtractor()

# Fetch a URL
print(extractor.process_query("https://example.com"))

# Extract information
print(extractor.process_query("Extract all the links from the webpage"))
print(extractor.process_query("Give me the main content of the webpage in JSON format"))
print(extractor.process_query("Provide a summary of the webpage content in CSV format"))
