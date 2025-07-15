import requests
import json
import logging
from typing import List, Dict, Any, Optional

class LlamaIntegration:
    """
    Clase para interactuar con el modelo Llama 3.1 a través de Ollama.
    """

    def __init__ (self,
                  host: str = "http://localhost",
                  port: int = 11434,
                  model_name: str = "llama3.1",
                  temperature: float = 0.7,
                  max_tokens: int = 200):
        
        self.base_url = f"{host}:{port}"
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def generate_response(self, context: List[str], prompt: str) -> str:
        """
        Genera una respuesta usando Llama 3.1 a través de Ollama.
        
        Args:
            context: Lista de mensajes previos en la conversación
            prompt: prompt específico para la generación

        Returns:
            La respuesta generada por el modelo
        """

        try:
            #Para construir la URL para la API de Ollama
            api_url = f"{self.base_url}/api/generate"

            #Construit el historial de mensajes para Llama 3
            conversation_history = "\n".join(context) if context else ""

            #El prompt final es el historial de la conversación más el prompt específico
            full_prompt = f"{conversation_history}\n{prompt}\nSputnik:"

            #Esto prepara los parámetros para la solicitud
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }

            self.logger.info(f"Enviando solicitud a Ollama con prompt: {full_prompt[:100]}...") # Loguea solo los primeros 100 caracteres del prompt

            #Para realizar la solicitud a Ollama:
            response = requests.post(api_url, json=payload, timeout=60) #Revisar si params es correcto o si debo añadir payload.

            #Verifica si la respuesta es exitosa
            if response.status_code == 200:
                response_data = response.json()
                generated_text = response_data.get("response", "")
                self.logger.info(f"Respuesta Generada correctamente: {generated_text[:100]}...")
                #Devuelve el texto generado por el modelo
                return generated_text
            else:
                self.logger.error(f"Error al llamar a Ollama: {response.status_code} - {response.text}")
                return "Lo siento, estoy teniendo problemas para procesar esa información."
            
        except Exception as e:
            self.logger.error(f"Excepción al generar respuesta: {str(e)}")
            return "Lo siento, estoy teniendo problemas para procesar esa información."

    def is_available(self) -> bool:
        """
        Función para verificar que el servicio de Ollama está dosponible
        Devuelve True si el servicio está disponible, False en caso contrario.
        """

        try:
            #Intentar hacer una solicitud sencilla para verificar disponibilidad:
            api_url = f"{self.base_url}/api/tags"
            response = requests.get(api_url, timeout=5) #Timeout de 10 segundos
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Error al verificar disponibilidad de Ollama: {str(e)}")
            return False
        
    def list_available_models(self) -> List[str]:
        """
        Obitne una lista de los modelos disponibles en Ollama.
        Devuelve una lista de nombres de modelos.
        """
        try:
            api_url = f"{self.base_url}/models"
            response = requests.get(api_url, timeout=10) #Timeout de 10 segundos
            
            if response.status_code == 200:
                models_data = response.json()
                models = [model["name"] for model in models_data.get("models", [])]
                return models
          
            else:
                self.logger.error(f"Error al obtener modelos disponibles: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            self.logger.error(f"Excepción al listar modelos disponibles: {str(e)}")
            return []