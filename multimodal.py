# Import necessary libraries for multi-modal processing and agent collaboration
from transformers import GPT4Tokenizer, GPT4LMHeadModel, CLIPProcessor, CLIPModel
from PIL import Image
import json
import os
import logging
import asyncio
import speech_recognition as sr
from functools import lru_cache

# Configure logging with debug level for more granular information
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Agent Registry to dynamically register and handle different input types
class AgentRegistry:
    def __init__(self):
        self.registry = {}

    def register_agent(self, input_type, agent):
        """
        Registers an agent for a specific input type (e.g., text, image, audio).
        """
        if input_type in self.registry:
            raise ValueError(f"Agent for input type '{input_type}' already exists.")
        self.registry[input_type] = agent
        logging.info(f"Registered {input_type} agent.")

    def get_agent(self, input_type):
        """
        Retrieves an agent based on input type.
        """
        return self.registry.get(input_type, None)

    def supported_types(self):
        """
        Returns the list of supported input types.
        """
        return list(self.registry.keys())


# Cognitive Architecture with agent registry for handling multi-modal inputs
class CognitiveArchitecture:
    def __init__(self, model_name="gpt4"):
        """
        Initializes the cognitive architecture with multi-modal processing capabilities.
        Uses an AgentRegistry for dynamic agent management.
        """
        self.agent_registry = AgentRegistry()
        self.short_term_memory = ShortTermMemory(max_cache_size=50)
        self.long_term_memory = LongTermMemory("long_term_memory.json")

        # Register agents for text, image, and audio processing
        self.agent_registry.register_agent("text", TextAgent(model_name))
        self.agent_registry.register_agent("image", ImageAgent())
        self.agent_registry.register_agent("audio", AudioAgent())
        logging.info("CognitiveArchitecture with multi-modal agents initialized.")

    async def process_input(self, input_data, input_type):
        """
        Processes input data by delegating to the appropriate agent based on input type.
        Includes validation for input type.
        """
        agent = self.agent_registry.get_agent(input_type)
        if not agent:
            return f"Unsupported input type: {input_type}. Supported types are: {', '.join(self.agent_registry.supported_types())}"
        
        # Process input using the relevant agent
        return await agent.process(input_data)


# Text Agent for handling text-based reasoning with caching
class TextAgent:
    def __init__(self, model_name="gpt4"):
        """
        TextAgent handles reasoning and natural language processing using GPT-4.
        """
        self.tokenizer = GPT4Tokenizer.from_pretrained(model_name)
        self.model = GPT4LMHeadModel.from_pretrained(model_name)
        logging.info(f"TextAgent with model {model_name} initialized.")

    @lru_cache(maxsize=100)  # Cache recent queries to optimize performance
    async def process(self, input_text):
        """
        Processes text reasoning with GPT-4. Uses caching to optimize repeated queries.
        """
        try:
            logging.debug(f"Processing text input: {input_text}")
            inputs = self.tokenizer.encode(input_text, return_tensors="pt")
            outputs = await asyncio.to_thread(self.model.generate, inputs, max_length=100, num_return_sequences=1)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logging.info(f"Generated text response: {response}")
            return response
        except Exception as e:
            logging.error(f"Error in TextAgent: {e}")
            return f"Error in processing text input: {e}"


# Image Agent for handling image processing with validation
class ImageAgent:
    def __init__(self):
        """
        ImageAgent uses CLIP (Contrastive Language-Image Pre-training) for image recognition.
        """
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logging.info("ImageAgent initialized with CLIP model.")

    async def process(self, image_path):
        """
        Processes an image for recognition. Validates the file path before proceeding.
        """
        if not os.path.exists(image_path):
            logging.error(f"Image file not found: {image_path}")
            return f"Error: Image file '{image_path}' not found."
        
        try:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = await asyncio.to_thread(self.model.forward, **inputs)
            logging.info(f"Processed image at {image_path}")
            return "Image processed successfully."  # For demo; Future: Map to human-readable output
        except Exception as e:
            logging.error(f"Error in ImageAgent: {e}")
            return f"Error processing image: {e}"


# Audio Agent for handling audio input with speech recognition
class AudioAgent:
    def __init__(self):
        """
        AudioAgent uses SpeechRecognition for speech-to-text conversion.
        """
        self.recognizer = sr.Recognizer()
        logging.info("AudioAgent initialized for speech-to-text processing.")

    async def process(self, audio_file):
        """
        Converts speech from audio file to text using SpeechRecognition.
        Validates audio file path before processing.
        """
        if not os.path.exists(audio_file):
            logging.error(f"Audio file not found: {audio_file}")
            return f"Error: Audio file '{audio_file}' not found."

        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                logging.info(f"Processed audio and recognized text: {text}")
                return text
        except Exception as e:
            logging.error(f"Error in AudioAgent: {e}")
            return f"Error processing audio: {e}"


# Memory classes remain the same from previous version


# Example usage of multi-modal input handling system with improved agents

async def main():
    # Initialize the cognitive system with dynamic agent registration
    cognitive_system = CognitiveArchitecture()

    # Example 1: Process text input
    print("Processing text input...")
    text_input = "What is the capital of France?"
    response = await cognitive_system.process_input(text_input, "text")
    print(response)

    # Example 2: Process image input (replace 'path_to_image' with actual image file path)
    print("\nProcessing image input...")
    image_input = "path_to_image.jpg"
    response = await cognitive_system.process_input(image_input, "image")
    print(response)

    # Example 3: Process audio input (replace 'path_to_audio' with actual audio file path)
    print("\nProcessing audio input...")
    audio_input = "path_to_audio.wav"
    response = await cognitive_system.process_input(audio_input, "audio")
    print(response)


# Entry point of the script
if __name__ == "__main__":
    asyncio.run(main())  # Start the multi-modal cognitive system
