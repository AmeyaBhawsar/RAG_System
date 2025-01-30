import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import ollama

# Load environment variables from .env file
load_dotenv()

# System Prompt for LLMs
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

Context will be passed as "Context:"
User question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

class LLMHandler:
    def __init__(self, model_name):
        self.model_name = model_name

        # Define available models inside the class
        self.groq_models = {
            "mixtral-8x7b-32768",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "gemma2-9b-it",
        }

        self.ollama_models = {
            "llama3.2:3b",  # The model you have installed
        }

        # Set API key only if using Groq
        self.api_key = os.getenv("GROQ_API_KEY") if self.model_name in self.groq_models else None

    def _stream_response(self, text: str, chunk_size: int = 20):
        """
        Simulate streaming by breaking the text into chunks and yielding them.
        """
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]

    def call_llm(self, context: str, prompt: str):
        """Call the appropriate LLM based on the selected model."""
        if self.model_name in self.groq_models:
            # Handle Groq API models
            llm = ChatGroq(model=self.model_name, api_key=self.api_key)
            response = llm.invoke(f"Context: {context}\nQuestion: {prompt}")
            
            # Simulate streaming for Groq models
            for chunk in self._stream_response(response.content):
                yield chunk

        elif self.model_name in self.ollama_models:
            # Handle Ollama local models
            response = ollama.chat(
                model=self.model_name,
                stream=True,  # Enable streaming
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context: {context}\nQuestion: {prompt}"},
                ],
            )
            for chunk in response:
                if chunk["done"] is False:
                    yield chunk["message"]["content"]
                else:
                    break
        else:
            raise ValueError(f"Model '{self.model_name}' not found in Groq or Ollama.")
